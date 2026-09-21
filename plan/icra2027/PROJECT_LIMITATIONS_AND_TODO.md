# SimAnyRoom：实验限制、项目状态与总体 TODO

核对日期：2026-09-06 UTC。Owner：root orchestrator。本文为项目审计，不是新实验 freeze，也不改变原始指标、任务分母或门槛。

**结论：当前已完成“现有结果的完整归档与论文发布”，尚未完成项目全部实验。最大的科学缺口是 E4：没有合格操作任务，也没有执行 policy rollout。论文已有可核验的 8 页版本，但最终科学提交 gate 仍为 FAIL。**

## 1. 当前版本与运行状态

| 项目 | 核对结果 |
|---|---|
| 本次诊断代码基线 | `83efa8e897bbb49a4d7726c9a54982a4d66ebaf2`，`/group/worldcept/code/SimAny-wt/icra-main`，branch `main` |
| 原始产物 checkout | `402cda047b28ac17f50cb7406fe82fcdebacaaba`，`/group/worldcept/code/SimAny`；本轮没有修改 |
| 论文 | `83c97b128813e9e382ac66927926758771f1d4af`，`/group/worldcept/code/SimAnyRoom` |
| 当前论文 publication freeze | `20260906-8cae8a0-v1`，producer `93c9566ac7fa76d73598f0394b8ae342257bf3db` |
| E4 全队列审计 freeze | `20260906-b9bdcf3-v1`，producer `05184e11874ff47fd22d6aa3e39f5464fe03b09c` |
| E4 原始构建/导出 | `20260906-0e7579a-v1`，producer `553db0575241a6858921b6d648d24c29392161fd` |
| 当前 SimAny Slurm jobs | 本轮 `squeue -u runyi_yang` 核对：无运行或排队的本项目 job；用户的其他项目 job 不计入 |
| 本轮计算 | CPU 只读审计，以及一个固定场景 A0/A4 的原始导出重放；没有提交 Slurm/GPU job，没有新增 policy episode |
| 共同最终实验 freeze | 尚不存在。当前 publication 绑定多个不可改写的阶段产物，不能声称全部 full 实验已在同一最终 freeze 完成 |

历史最终收尾 job：E1 `835404`、E4 `835518`、E2 `835594` 均已 COMPLETED；没有必要重新提交这些已完成任务。E4 `835518` 是终态证据审计，不是机器人操作任务。

## 2. 逐任务完成范围与缺口

| 任务 | 已完成的真实范围 | 未完成 / 限制 | 当前科学结论 |
|---|---|---|---|
| E0 共用契约 | freeze、哈希、不可覆盖、单治疗轴、来源验证与 preflight 已投入使用 | 每次源码/config 改动仍需新 freeze 和对应检查；旧 PASS 不自动覆盖新代码 | 工程契约 PASS，不代表实验主张 PASS |
| E1 构建规模 | 当前自动发现+splat-fused 范围：50 场景，1,871 输入，399 controller accepts，394 verified exports；独立 isolated-drop 稳定 361/394；44 成功导出场景、4 空、2 拒绝 | 五档输入阶梯仅有一档当前证据；其余 200/250 scene-regime cells NOT_RUN；导出物体 F1 与完整 runtime 未绑定 | 当前范围可报告；完整 Table I 未完成 |
| E2 保真度 | 50 场景/400 预定视图；raw 400/400，composite 320/400；共同 320 视图/40 场景比较已完成 | GT-discovery room row、Harmonizer row 与四个 object-method row 均未完成；10 场景缺失保持分母 | 当前 composite 平均外观改善主张 FAIL |
| E3 Agentic A0–A4 | 50 场景、1,871 jobs、9,355 policy rows；初始池共享；1,462 个真实替代注册动作重试；580 matched / 1,291 unmatched | A4 接受后质量不等于全总体优势；E3 isolated acceptance probe 不能当作 E4 房间稳定性或操作结果 | 窄范围 evidence selection/registration retry 主张 PASS；全面 A4 优越主张 FAIL |
| TRELLIS.2 | mesh-only 集成及独立 pilot：15/15 生成，1/15 isolated stable；3 个 geometry-matched、12 个未匹配 | 不属于当前 A0–A4 冻结初始池；不能后补进去使比较多变一个轴；不代表完整可模拟对象生成 | pilot 证据，不能升级为完整方法胜出 |
| E4 操作评测 | 完成全 50 场景前置条件与失败分母审计，见第 3 节 | 0 qualified tasks、0 实际 900-step qualification、0 policy episodes；reset bank/完整 treatment rollout 均未执行 | applicability FAIL；manipulation success NOT_RUN/null |
| E5 Harmonizer Option C | 协议、robot restore、五条件 coverage/指标管线及合成测试已实现；两份 Harmonizer 权重本地存在 | Cosmos 基座/tokenizer 权限 403；真实五条件输出、保留性、temporal LPIPS 与 p95 latency 未测 | 真实效果 NOT_RUN |
| E6 Task-local LOSO | 72 查询及缺失标志保留，16 constructor-failed；全部 72 缺 robot/camera 支持并被判 invalid | 六个 LOSO fold 都只有单类；禁止伪造类别、调整标签或报告不存在的预测指标 | 当前 applicability FAIL；AUROC/AUPRC/Brier/Risk NOT_RUN |
| E7 Real capture | 当前严格 TRAIN-only 协议保持 10 个 DROID episodes/9 labs；CPU 5 完成、5 失败；5 个后续 GS 阶段完成 | 后续 empty/zero-discovery 不能算严格 full build；phone 尚无合格的至少 3 房间预声明集。旧历史构建不能冒充新协议完成 | 阶段证据可报告；完整 real-world construction 未完成 |
| E8 真机匹配 | 已定义匹配/导入与缺失处理 | 无可用的真实匹配 physical trials/hardware；不能用 replay 代理真机成功 | NOT_RUN；physical-success 单元空，sim-real 主张移除/收窄 |
| E9 论文 | 39 生成产物、9 LaTeX tables、22 JSON/CSV sources、4 figures、4 audit files；337 数字源字段与22 claim sentences 核验；PDF 8 页、字体/引用/版面 QA 与便携审计包完成 | 后续新证据需重新生成；科学缺口和最终共用 freeze 仍未解决 | 当前 available-result publication COMPLETE；scientific submission FAIL |
| D0 演示 | live fallback 框架、3 段 6 秒 evidence clips、poster/subtitles 已有 | 缺真实同步操作与验证过的 Harmonizer 素材；最终 80–95 秒 hero、30 秒 teaser、8 秒 loop 未完成 | 最终演示 NOT_RUN |

E2 共同支持集上的冻结描述均值：raw PSNR `21.581642249757337` / SSIM `0.8444029330571121` / LPIPS `0.2993210877291858`；composite `20.437798054410234` / `0.8318254927240997` / `0.3167900979751721`。三项均为变差方向；不是显著性结论，不能隐藏。40 个 composite 场景包含 4 个 zero-object/no-removal 背景，不等于 40 个可交互房间。

E3 accepts：A0 `1640`、A1 `1800`、A2 `1800`、A3 `1800`、A4 `399`，共同分母均为 `1871`。A4 的 `399/399` isolated acceptance stability 是同一接受探针，不能与 E1 的独立 `361/394` 或 E4 room settle 混用。

## 3. E4 是什么、为什么目前没有 policy job

E4 使用唯一的 `robo.eval.harness_runner`，在固定 policy/checkpoint、机器人、相机、控制器/action convention、reset、horizon、instruction、rubric、seed 下，分别比较：

1. 构建：fixed single-path vs full agentic。
2. 碰撞：private support shims vs full-room collision。
3. 观测：MuJoCo raster vs raw composite vs Harmonizer+robot restoration。

任务包含 object-to-receptacle 与 object-to-region。三个比较块各自只改变一个声明的 treatment axis；需要分别计 coverage、success、grasp/lift/place、failure 与 latency。

### 已冻结的失败分母

- 50 场景、1,871 objects；6,155 输入查询，5,886 预声明 budget exclusions，269 selected queries。
- 2,690 **逻辑 qualification cells** = 805 endpoint unavailable + 865 construction-requirement failed + 1,020 source-unavailable NOT_RUN。
- 1,020 NOT_RUN = 970 planning unavailable + 50 export rejected。
- 场景终态：23 完成前置检查、16 planning unavailable、2 export rejected、9 original source no queries。
- 同一上游检查重复写入 5 个预定 reset cell，不是进行了 5 次物理或 policy trial。policy success 是 `null`，不是 `0%`。

### 本轮新诊断：仅诊断，不新增论文结果

| 问题 | 核对证据 | 分类与解释 |
|---|---|---|
| Endpoint 缺失 | 805 cells =161 policy-tasks。A0 110 cells 来自14独立对象 `trellis_missing → reject`；A4 695 cells 来自99独立对象 abstain，其中96 retry exhausted | 构建不可用/真实弃权。核对2,452 materialization roster 和1,302 accepted selected records，零状态映射不一致 |
| Target 漂移 | 860 cells =172 policy-tasks =106不同 scene-policy-target。全部为有限、非负且 ≥0.03m 的原始测量，0 missing/invalid；独立目标中位数1.099327m，范围0.048059–49.634664m | 真实导出物理不稳定。不是缺失值被统一标成 drift fail；不同 task 复用同一目标，不重复充当独立样本 |
| 容器角色 | 325 cells =65 policy-tasks /56独立 policy-receptacles；全部 drift ≥0.10m，其中29任务另有尺寸超限；无 label/eligibility failure | 当前实际阻塞是物理/尺寸；此阶段未执行 cavity 验证，不能称 cavity 失败 |
| 13 场景无尺寸合格目标 | 106/106 prepared semantic targets 的 discovery AABB 最大尺寸 >0.28m | 输入发现范围/任务适用性限制；未发现 mm/m 转换错误。不能看生成结果后重选尺寸合格目标 |
| 3 场景无桌面候选 | 2 场景目标高于冻结1.4m上限；`fb5a96b1a2` 的3个合格 book 因相邻高度链式聚类后的均值偏移被丢失，canonical planner 可复现 | 前2个为冻结任务范围限制；后1个为已定位 planner 候选遗漏，需要合成回归测试与新协议修复；不保证修后物理合格 |
| 2 export rejected | 每场景1个支撑 primitive 侵入大尺度 A0 物体；scale/observation 6.0583、3.3285 | 构建/碰撞有效性失败；尚未证明碰撞代码有误，不放松 carve guard |
| 9 原始无任务场景 | 138 objects 中2场景有3个 `computer mouse`；白名单只有 `mouse` | 明确标签 alias 缺口；可在新协议规范化，不能修改旧分母，也不证明这3个鼠标可操作 |

**原样重放验证**：按字典序选择首个有 canonical requirement failure 的场景 `09c1414f1b`，对 A0/A4 原始 XML 各执行1000步、timestep0.002s，未改变几何或阈值。两者最终 `state_hash` 与冻结报告完全相同，所有物体 drift 差值为0。例：A0 `obj_1003` 下落1.182455m且侧向移动；A4 `obj_1105` 首次接触是floor，下落0.955864m；另有初始物体穿插。由此支持优先检查支撑/位姿/尺度与完整房间碰撞。尚不能把全队列失败单独归因于 carve、某一个模型或某一个代码错误。

诊断脚本首轮遇到 source-unavailable 行的 `qualification_evidence=null`，已修复诊断读取并保留失败日志；原始 evaluator、freeze 和指标未改。重放是原有 export-settle 的复核，不增加 E4 实际900-step资格检查或 policy episode 计数。

## 4. 不可违反的实验限制

| 限制 | 执行要求 |
|---|---|
| 冻结与复用 | 全量来自 clean main commit、不可变config、E0 READY；源码/config变更用新freeze。复用需 source/input/object/config/checkpoint/camera/controller/metric/treatment 相关哈希一致；不得覆盖旧结果 |
| Ground truth 隔离 | 构建/选择/retry 不读 evaluation GT；生成视图、注册表面与评估视图/表面独立；hidden GT 仅在构建冻结后进入评估 |
| Coverage 先于质量 | reject、abstain、build failure、crash、timeout、safety stop、enhancer failure 全部留在声明分母；matched geometry 不冒充所有 discovered instances |
| 公平配对 | 一个比较仅改变一个 treatment axis；其他policy、硬件配置、相机、控制、reset和seed等冻结；A1–A4共享初始proposal池 |
| 真正 retry | 新proposal/artifact或真实不同动作，归档原因和依赖；重读旧资产不算重试。TRELLIS.2加入未来池须重新声明、配对和冻结 |
| 禁止结果驱动调参 | 不看full结果后换场景/对象/视图、放宽尺寸/漂移/标签门槛来凑正结果；开发修复只能新协议验证，并承认已观察测试集的边界 |
| Tier admission | smoke含负测例 → 真实pilot → 完整声明矩阵。失败只续跑缺失单元；工程smoke/诊断PASS不授权科学promotion |
| 负结果 | 保留geometry改善而稳定性下降、composite均值变差、E4前置失败和E6单类；科学失败不能被写成环境问题 |
| 既有 producers | 继续用 construction_metrics、fidelity_metrics、harness_runner、harmony_visual_metrics、audit_loso/audit_metrics、paper_pipeline；不另造rollout/指标管线 |
| Harmonizer | 不可失败后静默退回raw；temporal history按run/treatment/scene/task/reset/camera隔离；验证robot-core字节相等、masks、真实端到端p95与coverage；不声称几何/物理改善 |
| 资源与调度 | 用户最新要求：不使用job array；普通独立jobs。只对已经通过门槛的任务优先安排Hala、gcp*、sof1*空卡；CPU聚合不占GPU；不因空卡存在而提交无效policy任务 |
| 论文与demo | 数值全部经JSON/CSV和paper_pipeline生成；缺测保持空/NOT_RUN并收窄claim。最终视频必须真实selection/retry、连续同一policy episode、与最终E9数字同源 |

## 5. 外部阻塞：哪些需要额外资源

### Cosmos：需要，但它属于 E5，不阻塞当前 E4 原因诊断

- 精确repo：`nvidia/Cosmos-Predict2-0.6B-Text2Image`。
- 固定revision：`dd55b6858b22ad569976bff207880b8fea839da7`。
- 缺少：`config.json`、`model.pt`、`tokenizer/tokenizer.pth`。
- 预期目录：`/group/worldcept/code/SimAny/checkpoints/harmonizer/nvidia/Cosmos-Predict2-0.6B-Text2Image/`。
- 本地已有：`diffusion_harmonizer.pkl`、`harmonizer_nontemporal.pt`；它们不能替代上述基座/tokenizer。
- 本轮用现有集群身份进行metadata-only访问：`2026-09-06T12:06:23Z`，仍为HTTP403 / `GatedRepoError`。未下载、未输出凭据。
- 解除方式：提供已授权的该revision本地权重路径，或让集群当前Hugging Face身份获得仓库访问权限。不要在聊天中发送token。之后仍需哈希闭包、真实smoke/pilot与五条件评估。

其他外部缺口：预声明的真实phone房间集（至少3rooms）；E8真正匹配的physical trials和硬件。E1/E2尚未实现或尚未绑定的输入/指标主要是**本地实验工作**，不能笼统归为“缺数据/缺模型”。

## 6. 总体 TODO：按关键路径排列

- [x] 当前可用full结果归档、生成表、收窄claim、8页论文QA与便携数字核验包。
- [x] E4 失败分母分解、endpoint映射核对、全部目标漂移读取、固定场景原样导出重放、planner失败复现。
- [ ] **P0 / E4：**补充支撑、初始穿插、尺度和坐标的受控原因验证；确认哪一处是可修复代码/资产问题，哪一处是真实构建能力失败。对已定位alias/聚类遗漏加合成回归与负测例，修复进入新source/config/freeze，旧任务分母保持不动。
- [ ] **P0 / E4：**预声明独立开发pilot及失败判定。README紧凑范围：2rooms、每room2tasks（尽可能覆盖两任务族）、每block2treatments、5matched resets；先scripted scorer验证，再一个real-policy smoke。不能从当前full结果中仅挑稳定对象宣布成功。
- [ ] **P0 / E4：**pilot允许后建立canonical reset/rollout ledger，按构建、碰撞、观测分别补矩阵。完整目标≥4rooms、≥4有效tasks/room、条件允许≥2policycheckpoints、每arm≥60matched episodes（100preferred），单scene/task不超过35%。若无法实现，保持NOT_RUN并收窄下游claim。
- [ ] **P1 / E1+E2 CPU并行：**建立精确object/proposal/mesh/transform/reference哈希复用桥和完整runtime-stage receipts；检查580matched独立参考面能否合法复用，1291unmatched继续null。
- [ ] **P1 / E2：**固定共同对象/独立held-out view与surface roster；补TRELLIS、ReconViaGen、evidence-selected、evaluation-only oracle四行masked appearance/CD/F1/collapse。GT selector不能回流controller。
- [ ] **P1 / E5（等权限）：**补全固定Cosmos模型闭包，运行真实多camera/多episode smoke/pilot；五条件输出、保留性/temporal/latency；达到保留性门槛后才进入E4 observation arm。
- [ ] **P2 / E1：**补另外四档input regimes，逐一smoke/pilot/full；补当前导出体F1与完整runtime，不移植无法验证的历史50-room数字。
- [ ] **P2 / E2：**补GT-discovery room composite；按原声明视图保留失败coverage。
- [ ] **P2 / E6：**先恢复真实robot/camera/task support；新特征队列先冻结、后标签、再LOSO。当前72invalid结果不改；不为双类而挑样本。
- [ ] **P2 / E7：**保留10DROID attempts并补可执行strict full-build阶段；准备至少3个预声明phone rooms，报告alignment/runtime/failures。
- [ ] **条件任务 / E8：**仅有真实匹配physical trials时执行；否则明确NOT_RUN，空表格，移除sim-real成功claim，不阻塞E9可用证据发布。
- [ ] **D0：**取得真实连续同步policy和通过验证的Harmonizer素材；生成80–95秒hero、30秒teaser、8秒无缝loop、poster、字幕、source manifest；验证live不健康时≤3秒本地fallback。
- [ ] **最后 / E9：**冻结最终共同输出集合；canonical重新生成全部表和claim ledger；全部精确句子都有PASS/FAIL/NOT_RUN处理；≤8页含references、全页render、嵌入字体、无undefined/overfull；paper和code提交push。

DDL取舍：先E4原因验证，再有效pilot；并行做无需GPU的E1/E2复用桥。不要为追求正结论反复扫模型或放松full-test门槛。如果E4无法形成有效任务，就提交有充分证据的窄范围构建结论，并明确操作/真机部分未验证，不能把“best paper”作为可保证的结果。

## 7. 可复核路径与命令

以下相对路径均以 `/group/worldcept/code/SimAny-wt/icra-main` 为根；原始freeze位于 `/group/worldcept/code/SimAny/outputs/icra2027/`。

| 证据 | 路径 |
|---|---|
| 当前任务优先级审计 | `plan/icra2027/REMAINING_WORK.md` |
| 本轮端点诊断 | `outputs/e4-endpoint-diagnosis-20260906/{summary.json,endpoint_policy_tasks.json,role_policy_tasks.json,source_hashes.json,audit.py}` |
| 本轮规划诊断 | `outputs/icra2027/e4-planning-diagnostic-20260906T122500Z/{README.md,planning_diagnostic.json,canonical_planner_replay.json,audit.py}`；目录名为标签，时间以记录为准 |
| 本轮漂移/原样重放 | `outputs/e4-root-cause-diagnosis-20260906/{drift_summary.json,drift_rows.json,input_identities.json,exact_export_replay.json,diagnose.py,diagnose.log}` |
| Cosmos当前权限 | `outputs/e4-root-cause-diagnosis-20260906/cosmos_access.json` |
| E4冻结分母与终态 | `20260906-b9bdcf3-v1/full_qualification/{gate.json,scene_receipts.json,planned_qualification_cells.jsonl}`；上层 `completion_audit.json` |
| 当前E1输出 | `20260906-3e8ab6d-v1/construction/table/construction_table.json` |
| 当前E2输出 | `20260906-b689700-v1/fidelity/public_factorized/metrics/receipt.json` |
| 当前论文表/包 | `20260906-8cae8a0-v1/paper_tables/`；`SimAnyRoom-audit-83c97b1.tar.gz` |
| Paper QA | `/group/worldcept/code/SimAnyRoom/audit/paper_qa_full_results.json` |

本轮执行命令：

```bash
squeue -u runyi_yang -o '%.18i %.12P %.32j %.8T %.10M %.30R'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  /group/worldcept/code/SimAny/.venv/bin/python \
  outputs/e4-root-cause-diagnosis-20260906/diagnose.py
```

上述diagnose记录为本轮复核；再次执行应先复制脚本到新的审计目录，避免覆盖本轮审计记录。没有重新运行全局测试或paper build；本轮只有文档/诊断产物变更。此前E9 source的389 focused/130 exact E0 PASS和8页QA是该版本的已有证据，不冒充本轮重跑。
