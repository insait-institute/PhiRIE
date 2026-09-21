"""Validate and analyze genuinely matched simulation/physical trials."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from robo.eval.metric_utils import bootstrap_interval, write_csv, write_json


def _parse_binary(value, field):
    if value in {0, "0", False, "false", "False"}:
        return 0
    if value in {1, "1", True, "true", "True"}:
        return 1
    raise ValueError(f"{field} must be a recorded binary outcome, got {value!r}")


def load_pairs(path):
    rows = list(csv.DictReader(Path(path).open()))
    required = {"source", "scene_id", "task_id", "policy_id", "reset_id",
                "sim_success", "real_success"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"paired-trial file lacks columns {sorted(missing)}")
    keys = [(row["source"], row["scene_id"], row["task_id"],
             row["policy_id"], row["reset_id"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate paired trial IDs: {duplicates[:10]}")
    for row in rows:
        row["sim_success"] = _parse_binary(row["sim_success"], "sim_success")
        row["real_success"] = _parse_binary(row["real_success"], "real_success")
        row["audit_accept"] = str(row.get("audit_accept", "false")).lower() in {
            "1", "true", "yes"}
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], "all")].append(row)
        if row["audit_accept"]:
            groups[(row["source"], "audit_accepted")].append(row)
    output = []
    for (source, subset), values in groups.items():
        sim = np.asarray([row["sim_success"] for row in values], dtype=float)
        real = np.asarray([row["real_success"] for row in values], dtype=float)
        differences = np.abs(sim - real)
        agreement = 1.0 - differences
        gap_samples = sim - real
        gap_lo, gap_hi = bootstrap_interval(gap_samples, statistic=lambda x: abs(np.mean(x)))
        agr_lo, agr_hi = bootstrap_interval(agreement)
        output.append({
            "source": source, "subset": subset,
            "task_policy_pairs": len({(row["scene_id"], row["task_id"], row["policy_id"])
                                      for row in values}),
            "matched_trials": len(values),
            "sim_success": float(sim.mean()), "real_success": float(real.mean()),
            "absolute_success_gap": abs(float(sim.mean() - real.mean())),
            "gap_ci95_lo": gap_lo, "gap_ci95_hi": gap_hi,
            "outcome_agreement": float(agreement.mean()),
            "agreement_ci95_lo": agr_lo, "agreement_ci95_hi": agr_hi,
            "audit_coverage": len(values) / max(
                len(groups.get((source, "all"), values)), 1),
        })
    return output


def generate(input_csv, out_dir):
    rows = load_pairs(input_csv)
    summary = summarize(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "paired_real_world.json", {
        "rows": summary, "matched_trial_count": len(rows)})
    write_csv(out_dir / "paired_real_world.csv", summary)
    return {"rows": summary, "matched_trial_count": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.input, args.out), indent=2))


if __name__ == "__main__":
    main()
