"""Standard task-local audit metrics and LaTeX-table generation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from robo.eval.metric_utils import write_csv, write_json

LATEX_NEWLINE = r"\\"


def auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = labels == 1, labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        idx = np.flatnonzero(scores == value)
        ranks[idx] = ranks[idx].mean()
    rank_sum = ranks[positives].sum()
    return float((rank_sum - positives.sum() * (positives.sum() + 1) / 2)
                 / (positives.sum() * negatives.sum()))


def auprc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / labels.sum())


def risk_at_coverage(labels, invalid_probability, coverage: float) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(invalid_probability, dtype=float)
    keep = max(1, int(np.floor(len(labels) * coverage)))
    accepted = np.argsort(scores, kind="mergesort")[:keep]
    return float(labels[accepted].mean())


def compute(labels, scores) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    return {
        "n": int(len(labels)), "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
        "risk80": risk_at_coverage(labels, scores, 0.8),
        "risk60": risk_at_coverage(labels, scores, 0.6)}


def risk_coverage_curve(labels, scores) -> list[dict]:
    """Exact empirical selective-risk curve, including every accepted count."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    cumulative_invalid = np.cumsum(ordered_labels)
    return [{
        "accepted": int(index),
        "coverage": float(index / len(labels)),
        "max_invalid_probability": float(ordered_scores[index - 1]),
        "risk": float(cumulative_invalid[index - 1] / index),
    } for index in range(1, len(labels) + 1)]


def generate(input_csv: str | Path, out_dir: str | Path) -> dict:
    rows = list(csv.DictReader(Path(input_csv).open()))
    grouped = defaultdict(lambda: ([], []))
    for row in rows:
        grouped[row["signal"]][0].append(int(row["invalid_label"]))
        grouped[row["signal"]][1].append(float(row["invalid_probability"]))
    results = [{"signal": name, **compute(*values)}
               for name, values in grouped.items()]
    risk_rows = []
    for name, values in grouped.items():
        risk_rows.extend({"signal": name, **row}
                         for row in risk_coverage_curve(*values))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "audit_table.json", results)
    write_csv(out_dir / "audit_table.csv", results)
    write_csv(out_dir / "risk_coverage.csv", risk_rows)
    header = (
        r"Reliability signal & AUROC $\uparrow$ & AUPRC $\uparrow$ & "
        r"Brier $\downarrow$ & Risk@80 $\downarrow$ & Risk@60 $\downarrow$ "
        + LATEX_NEWLINE)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\textbf{Can local evidence identify unreliable task queries?} Evaluation uses hidden-ground-truth validity labels and leave-one-scene-out predictions. Failed builds remain invalid queries in the denominator.}",
        r"\label{tab:audit}", r"\centering\scriptsize",
        r"\begin{tabular}{lccccc}", r"\toprule", header, r"\midrule"]
    for row in results:
        label = row["signal"].replace("_", " ")
        lines.append(
            f"{label} & {row['auroc']:.3f} & {row['auprc']:.3f} & "
            f"{row['brier']:.3f} & {row['risk80']:.3f} & {row['risk60']:.3f} "
            + LATEX_NEWLINE)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "audit_table.tex").write_text("\n".join(lines))
    return {"rows": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.input, args.out), indent=2))


if __name__ == "__main__":
    main()
