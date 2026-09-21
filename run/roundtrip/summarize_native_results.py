#!/usr/bin/env python3
"""Render the existing native paper tables as a readable result handoff.

No metrics, episode labels, or uncertainty estimates are calculated here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(release, out, paper_commit):
    release=Path(release).resolve()
    def read(name):return json.loads((release/name).read_text())
    manifest=read('release_manifest.json')
    for name,digest in manifest['artifacts'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('release artifact changed: '+name)
    rows=read('T2_manipulation.json')
    if not rows or not all(r['complete'] for r in rows):
        raise ValueError('summary requires a complete declared native cohort')
    aliases={'REF_NATIVE':'REF 原生参考','B0_FIXED_NATIVE':'B0 固定构建',
             'B3_AGENT_NATIVE':'B3 选择与注册重试','B4_ROOM_REPAIR_NATIVE':'B4 上下文修复',
             'BM_BUDGET_MATCHED_NATIVE':'BM 固定顺序预算匹配'}
    lookup={r['method']:r for r in rows}
    ref=lookup['REF_NATIVE']
    pct=lambda value:'未测量' if value is None else f'{100*value:.2f}%'
    lines=['# 原生重建实验最终结果','',
           '本文件直接渲染 robo.eval.paper_pipeline 的 JSON 输出；不重新计算指标或修改实验标签。','',
           f'- Release：`{release}`',f'- 论文提交：`{paper_commit}`',
           f"- 主实验：{ref['instances']} 个独立实例、{ref['layouts']} 个布局、{manifest['planned_units']} 个计划单元；{manifest['measured_units']} 个有明确结局。",
           '- 范围：L0 仅替换目标，保留原生房间与目的地；不是全房间重建。','',
           '| 方法 | 执行/计划 | 成功/计划 | 总体成功率 | 执行后的条件成功率 |',
           '|---|---:|---:|---:|---:|']
    for method,label in aliases.items():
        r=lookup[method]
        lines.append(f"| {label} | {r['executed']}/{r['planned']} | {r['successes_observed']}/{r['planned']} | {pct(r['success_per_planned'])} | {pct(r['success_per_executed'])} |")
    lines+=['','构建失败与弃权保留在计划分母内；未执行单元的原生策略成功值为空，不能称为实际策略失败。',
            '', '## 配对比较','', '| 方法减去对照 | 差值（百分点） | 分层配对 95% 区间 |','|---|---:|---|']
    pairs={('B3_AGENT_NATIVE','B0_FIXED_NATIVE'),('B4_ROOM_REPAIR_NATIVE','B3_AGENT_NATIVE'),('B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE')}
    for r in read('comparison_statistics.json'):
        if (r['method'],r['reference']) not in pairs:continue
        if not r['paired_population_complete']:raise ValueError('incomplete paired comparison')
        lo,hi=r['ci95_pp']
        lines.append(f"| {aliases[r['method']]} − {aliases[r['reference']]} | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines+=['','这些是固定策略和有限布局下的描述性分层区间。选择/重试与固定构建的比较包含额外生成和计算，不能单独证明选择机制或单位计算收益；不显著不是保留能力的证据。','',
            '## 重建范围对照','', '| 方法 | 范围 | 实例/布局 | 执行/计划 | 成功/计划 |','|---|---|---:|---:|---:|']
    for r in read('T4_native_controls.json')+read('T4_scope.json'):
        if not r['complete']:raise ValueError('scope cohort incomplete')
        lines.append(f"| {aliases[r['method']]} | {r['scope']} | {r['instances']}/{r['layouts']} | {r['executed']}/{r['planned']} | {r['successes_observed']}/{r['planned']} |")
    lines+=['','该范围实验含独立进程中的新匹配控制，不能与主实验直接混合。L1 替换目标和目的地的直接盆体几何，保留原生框架、附件与任务区域；完整工作区 L2 仍未运行。','',
            '## 失败复核与证据范围','']
    for r in read('native_import_adjudication_summary.json'):
        lines.append(f"- {aliases[r['method']]} 的 {r['classified_units']} 个单元、{r['affected_instances']} 个实例，经冻结源码/环境/原始日志/资产哈希独立复核，将通用 CODE_FAILED 分类为构建导入失败。原始记录、资产、阈值与策略结果不变。构建生产器接受 {r['producer_accepted_objects']} 个资产，实际进入原生策略执行 {r['native_executed_instances']} 个实例。")
    lines+=['- 该碰撞资产封闭且体积为正，但局部非凸性略超冻结阈值；不能笼统称为破碎或开口网格。',
            '- 外观和几何指标见 T1a/T1b JSON，保留公共匹配支持与缺失分母。完整画面由原生背景主导，不代表全房间或 Gaussian 外观保留。',
            '- 原生成功判据仍受对象绑定/原点语义限制，不证明整个网格包含关系或绝对姿态对应。',
            '- GS 可见性未通过；完整 L2、全房间操作、确认性反馈收益与真实机器人匹配试验没有据此获得支持。','',
            '## 可复查产物','',
            '- 机器表格：`T2_manipulation.json`、`comparison_statistics.json`、`T4_scope.json`。',
            '- 声明与来源：`test_conclusion_ledger.json`、`paper_table_provenance.json`、`release_manifest.json`。',
            '- 复核来源：`native_import_adjudications.json` 与原始终态、日志和独立审计哈希。']
    video=release.parent/'native_demo'
    if (video/'qa.json').exists() and json.loads((video/'qa.json').read_text()).get('status')=='PASS':
        lines+=['',f'- 同一 release 的视频与 QA：`{video}`（90 秒、30 秒和 8 秒）。']
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--paper-commit',required=True)
    args=parser.parse_args();summarize(args.release,args.out,args.paper_commit)
