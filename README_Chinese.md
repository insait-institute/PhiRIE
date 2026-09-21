# PhiRIE

[English version](README.md)

> [PhiRIE v2.0.1 发布说明与验证报告](docs/releases/v2.0.1.md)：代码、版本与分支整理状态。下方历史实验结果不代表本次发布重新运行或完成了完整科学验证。

输入一间真实房间的带位姿 RGB 图像，全自动输出一个可编辑、照片级真实、
可直接用于仿真的数字孪生。

> GitHub 仓库名为 **PhiRoom**（[github.com/RunyiYang/PhiRIE](https://github.com/RunyiYang/PhiRIE)）；
> 系统在论文中的名称是 SimAny。

给定室内场景的带位姿 RGB 图像及其现成的重建结果（网格与 3D 高斯 splat），
SimAny 会发现场景中的每一个物体，用生成的 3D 资产替换它并配准回度量尺度的
场景中，为其标注碰撞几何与物理参数（URDF），并把它从背景 splat 中擦除——
因此物体移动后，孪生场景仍保持照片级真实。不需要语义标注、不需要逐物体
提示、不需要点击；输出无需任何真值即可完成验证。

## 系统能力

- 自动物体发现（SAM3）→ 图像到 3D 资产生成（TRELLIS / ReconViaGen 混合）
  → Sim(3) 配准 → 碰撞几何（CoACD）+ 物理参数 → 逐物体 URDF。
- 高斯原生的物体移除与背景补全，使编辑后画面保持照片级真实——包括
  几何选择器会漏掉的透明物体。
- 导出到 PyBullet、MuJoCo/MJCF、Isaac Lab 和 OmniGibson。
- 无真值全自动模式（`run/run_auto.sh`）与真值驱动的基准模式并存；
  两种模式下的下游阶段完全一致。
- 单图零样本模式：一帧图像 + 单目度量深度，无需重建。
- 在导出场景内进行闭环 pi0.5 机器人策略评测，观测为照片级真实的
  合成画面（[robo/](robo/)）。
- 交互式 viser 编辑器：拖动资产、重跑物理、回放观看
  （[interface/viewer.py](interface/viewer.py)）。

## 仓库结构

| 目录 | 内容 |
|---|---|
| [agents/](agents/) | 生成流水线：`core`（路径/IO/相机/词表）、`discover`、`assets`、`edit`（高斯原生 inpainting）、`render`、`eval`、`baselines`、`single_image` |
| [models/](models/) | 外部神经模型的可运行桥接：`s1_segment`（SAM3）、`s2_depth`、`s4_trellis`、`s4_reconviagen` |
| [robo/](robo/) | 机器人层：pi0.5 的 MuJoCo 环境/机械臂配置/任务/评测、照片级渲染、仿真器导出（`sim/`） |
| [interface/](interface/) | viser 多场景编辑器、MuJoCo 实时查看器、演示视频/会话 |
| [run/](run/) | `env.sh`（路径 + 环境变量开关）、环境安装、流水线启动脚本、slurm 作业 |
| [coding_agents/](coding_agents/) | AI 编程智能体的痕迹：memory、history、skills |
| [data/](data/) | 数据集（内容已 gitignore） |
| [checkpoints/](checkpoints/) | 下载的模型权重（已 gitignore） |
| [third_party/](third_party/) | 第三方代码：TRELLIS、ReconViaGen、MaskClustering、FlashSplat、BEHAVIOR-1K、mujoco_menagerie 等 |
| [docs/](docs/) | 文档与论文 LaTeX |
| [tests/](tests/) | 无需数据集的配准压力测试 |

## 快速开始

```bash
bash run/setup_env.sh        # installs the main .venv; two more envs are needed,
                             # see docs/ENVIRONMENTS.md

bash run/run_factory.sh      # GT-driven benchmark mode (one scene)
bash run/run_auto.sh         # fully automatic mode, no annotations
bash run/run_inpaint.sh      # Gaussian-native removal + background completion
bash run/run_simfoundry.sh   # SimFoundry 复现基线（单帧零样本）

# interactive editor (runs in the gsplat env, serves on :8090)
source run/env.sh
run_gs interface.viewer --outputs-root outputs
```

场景选择及所有路径/开关均通过环境变量控制（`SIMANY_SCENE`、`SIMANY_OUT`、
`SIMANY_AUTO=1` 等），记录在 [run/env.sh](run/env.sh) 与各 `run/run_*.sh`
启动脚本中。各阶段以
python 模块形式从仓库根目录运行（`python -m agents.assets.s5_align`），
或使用 `run/env.sh` 定义的 `run` / `run_sam3` / `run_gs` / `run_qwen`
辅助命令——三个 python 环境是不可避免的，原因见
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)。

## 文档

| 页面 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 带注释的代码地图、运行模型、环境变量参考、新旧路径迁移表 |
| [docs/PIPELINE.md](docs/PIPELINE.md) | 生成流水线逐阶段说明，四种运行模式 |
| [docs/ROBOT.md](docs/ROBOT.md) | pi0.5 闭环评测：组件、运行方法、真实结果 |
| [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md) | 数据集、模型权重、third_party 清单、outputs/ 结构 |
| [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) | 系统主张、经过审核的数字，以及刻意不做的主张 |
| [docs/BASELINES.md](docs/BASELINES.md) | 与同期系统的定位对比；哪些对比实验尚未完成 |
| [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) | 为什么需要三个 python 环境，各自包含什么 |
| [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) | 论文工作笔记 |
| [docs/PAPER_REVISIONS.md](docs/PAPER_REVISIONS.md) | 2026 年 7 月同期工作出现后需要做的修订 |
| [docs/DEMO_STORYBOARD.md](docs/DEMO_STORYBOARD.md) | 演示视频分镜脚本 |
| [docs/related/](docs/related/) | 所引用先前工作的原始资料 |
| [docs/paper/](docs/paper/) | 论文 LaTeX 源码与 `build.sh` |
| [coding_agents/](coding_agents/) | AI 编程智能体的 memory/history/skills 组织方式 |

## 命名说明

本项目以代号 *SimFoundry* 开发（最初是对 arXiv:2606.28276 的复现），
后短暂使用 *PhiRoom* 作为项目名。这两个项目名均已弃用：系统名为
**SimAny**，GitHub 仓库保留 **PhiRoom** 这一名称，而 **"SimFoundry"
现在仅指代我们引用并对比的 NVIDIA/Stanford 先前系统**。旧的 `SIMF_*`
环境变量前缀仍可读取（python 各阶段会给出弃用警告）。

## 主要结果

实验协议：除特别说明外，均为 ScanNet++ v2 全部 50 个验证场景
（完整表格与注意事项见 [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md)）。

- 几何 F1@20 mm：真值驱动模式 **0.783**（混合生成；单视角为 0.708），
  全自动模式 **0.582**（相对匹配上的真值物体计分）。
- 无任何标注的全自动模式：1,082 个实例，tier-A+B 产出率 62.0%，
  drop-test 稳定率 75.4%。
- 开销：真值驱动模式 50 个场景共 9.7 A6000 小时，全自动模式 14.1
  （每场景 11.6 / 17.0 分钟）。
- 官方 DSLR 测试划分上的外观：合成孪生 28.32 dB，SceneSplat 背景
  28.89 dB——差距 0.57 dB；全自动孪生 27.52 dB。
