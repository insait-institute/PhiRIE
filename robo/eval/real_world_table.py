"""Render real-world construction and paired manipulation summaries as LaTeX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robo.eval.real_world_metrics import generate as aggregate_real_world

LATEX_NEWLINE = r"\\"


def _fmt(value, digits=1, percent=False):
    if value is None or not np.isfinite(value):
        return "--"
    if percent:
        return f"{100.0 * float(value):.1f}\\%"
    return f"{float(value):.{digits}f}"


def render_latex(payload: dict) -> str:
    construction = payload.get("construction", [])
    trials = payload.get("trials", [])
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\textbf{Real-world evidence under the same frozen harness contract.} Part (a) reports all captured workspaces before task-specific filtering. Part (b) contains only task-policy pairs with matched physical trials; demonstration replay is never relabeled as autonomous real-world success.}",
        r"\label{tab:real-world}", r"\centering\scriptsize",
        r"\begin{minipage}[t]{0.48\textwidth}", r"\centering",
        r"\textbf{(a) Captured workspace construction and alignment}\\[2pt]",
        r"\setlength{\tabcolsep}{2.3pt}", r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Source & workspaces & full builds & align pass & cm $\downarrow$ & deg $\downarrow$ \\",
        r"\midrule",
    ]
    for row in construction:
        lines.append(
            f"{row['source'].replace('_', ' ')} & {row['workspaces']} & "
            f"{row['full_builds']} & {_fmt(row['alignment_pass_rate'], percent=True)} & "
            f"{_fmt(row['translation_cm'])} & {_fmt(row['rotation_deg'])} {LATEX_NEWLINE}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}",
              r"\hfill", r"\begin{minipage}[t]{0.50\textwidth}", r"\centering",
              r"\textbf{(b) Matched physical manipulation}\\[2pt]",
              r"\setlength{\tabcolsep}{2.0pt}", r"\begin{tabular}{llcccc}",
              r"\toprule",
              r"Source & subset & pairs & sim succ. & real succ. & $|\Delta|$ $\downarrow$ \\",
              r"\midrule"]
    for row in trials:
        lines.append(
            f"{row['source'].replace('_', ' ')} & {row['subset'].replace('_', ' ')} & "
            f"{row['task_pairs']} & {_fmt(row['sim_success'], percent=True)} & "
            f"{_fmt(row['real_success'], percent=True)} & "
            f"{_fmt(row['absolute_gap'], percent=True)} {LATEX_NEWLINE}")
    if not trials:
        lines.append(r"No matched trials yet & -- & -- & -- & -- & -- \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}",
              r"\end{table*}", ""]
    return "\n".join(lines)


def generate(construction_path, trials_path, out_dir):
    payload = aggregate_real_world(construction_path, trials_path, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "real_world_table.tex").write_text(render_latex(payload))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--trials")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.construction, args.trials, args.out), indent=2))


if __name__ == "__main__":
    main()
