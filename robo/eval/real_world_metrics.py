"""Generate real-workspace construction/alignment and paired-trial summaries."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from robo.eval.metric_utils import write_csv, write_json


def _read(path):
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        return value if isinstance(value, list) else value.get("rows", [])
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return list(csv.DictReader(path.open()))


def _truth(value) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def summarize_construction(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (str(row["source"]), str(row.get("workspace_id")))
        if identity in seen:
            raise ValueError("duplicate real-world workspace")
        seen.add(identity)
        grouped[str(row["source"])].append(row)
    results = []
    for source, values in grouped.items():
        n = len(values)
        translations = [float(v["translation_cm"]) for v in values
                        if v.get("translation_cm") not in {None, ""}]
        rotations = [float(v["rotation_deg"]) for v in values
                     if v.get("rotation_deg") not in {None, ""}]
        results.append({
            "source": source, "workspaces": n,
            "full_builds": sum(_truth(v.get("full_build_success", v.get("full_build"))) for v in values),
            "reconstructions": sum(_truth(v.get("reconstruction_success")) for v in values),
            "reconstruction_coverage": sum(_truth(v.get("reconstruction_success")) for v in values) / n,
            "full_build_coverage": sum(_truth(v.get("full_build_success", v.get("full_build"))) for v in values) / n,
            "accepted_objects": sum(int(v["accepted_objects"]) for v in values if v.get("accepted_objects") not in {None, ""}),
            "accepted_object_records": sum(v.get("accepted_objects") not in {None, ""} for v in values),
            "alignment_evaluated": sum(v.get("alignment_pass") not in {None, ""} for v in values),
            "runtime_records": sum(v.get("runtime_minutes") not in {None, ""} for v in values),
            "runtime_minutes": (float(np.mean([float(v["runtime_minutes"]) for v in values if v.get("runtime_minutes") not in {None, ""}]))
                                if any(v.get("runtime_minutes") not in {None, ""} for v in values) else None),
            "failure_reasons": dict(Counter(v.get("failure_reason") for v in values if v.get("failure_reason"))),
            "alignment_pass_rate": (sum(_truth(v.get("alignment_pass"))
                                       for v in values) / max(n, 1)
                                    if any(v.get("alignment_pass") not in {None, ""} for v in values) else None),
            "translation_cm": float(np.mean(translations)) if translations else None,
            "rotation_deg": float(np.mean(rotations)) if rotations else None,
            "queries": sum(int(v.get("queries", 0)) for v in values)})
    return results


def summarize_trials(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row.get("subset", "all")))].append(row)
    results = []
    for (source, subset), values in grouped.items():
        pairs = {(v.get("scene_id"), v.get("task_id"), v.get("policy_id"))
                 for v in values}
        sim_values = [int(v["sim_success"]) for v in values
                      if v.get("sim_success") not in {None, ""}]
        real_values = [int(v["real_success"]) for v in values
                       if v.get("real_success") not in {None, ""}]
        results.append({
            "source": source, "subset": subset, "task_pairs": len(pairs),
            "sim_trials": len(sim_values), "real_trials": len(real_values),
            "sim_success": float(np.mean(sim_values)) if sim_values else None,
            "real_success": float(np.mean(real_values)) if real_values else None,
            "absolute_gap": (abs(float(np.mean(sim_values)) - float(np.mean(real_values)))
                             if sim_values and real_values else None),
            "audit_coverage": sum(_truth(v.get("audit_accept")) for v in values)
                              / max(len(values), 1)})
    return results


def generate(construction_path, trials_path, out_dir):
    records = _read(construction_path)
    construction = summarize_construction(records)
    trials = summarize_trials(_read(trials_path)) if trials_path else []
    out_dir = Path(out_dir)
    payload = {"construction": construction, "trials": trials}
    if any("paper_ready" in row for row in records):
        payload["paper_ready"] = all(row.get("paper_ready") is True for row in records)
    write_json(out_dir / "real_world_table.json", payload)
    write_csv(out_dir / "real_world_construction.csv", construction)
    write_csv(out_dir / "real_world_trials.csv", trials)
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
