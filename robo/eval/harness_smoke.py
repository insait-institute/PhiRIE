"""Dependency-light end-to-end smoke test for the paired-harness data plane.

It does not claim robot results. It produces a complete synthetic reset/treatment
ledger, validates paired coverage, and generates the exact main-table artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robo.eval.harness_spec import load_harness_spec
from robo.eval.harness_validation import validate_records
from robo.eval.main_table import generate as generate_main_table


def run(config_path: str | Path, out_dir: str | Path, resets: int = 3):
    config_path, out_dir = Path(config_path), Path(out_dir)
    config = json.loads(config_path.read_text()) if config_path.suffix == ".json" else None
    if config is None:
        import yaml
        config = yaml.safe_load(config_path.read_text())
    spec = load_harness_spec(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "harness_ledger.jsonl"
    records = []
    for treatment_index, treatment in enumerate(spec.treatments.values()):
        for reset_index in range(resets):
            success = (reset_index + treatment_index) % 3 == 0
            records.append({
                "episode_id": f"{treatment.id}__smoke_{reset_index}",
                "treatment_id": treatment.id,
                "scene_id": "synthetic_smoke", "task_id": f"task_{reset_index}",
                "task_family": ("object_to_receptacle" if reset_index % 2 == 0
                                else "object_to_region"),
                "reset_state_id": f"smoke_{reset_index}", "base_seed": 0,
                "reset_seed": reset_index,
                "outcome": "success" if success else "task_failure",
                "success": success, "score": 1.0 if success else 0.25,
                "stages": {"grasp": True, "lift": success,
                           "hover": success, "place": success},
                "ticks": 1, "wall_s": 0.001, "error": None})
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    reset_ids = {f"smoke_{index}" for index in range(resets)}
    validation = validate_records(records, spec, reset_ids)
    if not validation["ok"]:
        raise RuntimeError(validation["violations"])
    table = generate_main_table(config, ledger_path, out_dir / "paper_tables")
    summary = {"validation": validation, "table": table}
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resets", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.out, args.resets), indent=2))


if __name__ == "__main__":
    main()
