"""Create predeclared hidden-ground-truth task-validity labels.

The label is invalid when any required task-local invariant fails. Missing
required evidence is also invalid; it is never imputed as a pass.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from robo.eval.metric_utils import write_csv, write_json


@dataclass(frozen=True)
class ValidityThresholds:
    translation_cm: float = 5.0
    rotation_deg: float = 15.0
    scale_error_pct: float = 15.0
    penetration_cm: float = 1.0


def _bool(value):
    if isinstance(value, bool):
        return value
    if value in {1, "1", "true", "True", "yes", "YES"}:
        return True
    if value in {0, "0", "false", "False", "no", "NO"}:
        return False
    return None


def _float(value):
    if value in {None, "", "--"}:
        return None
    return float(value)


def label_record(record: dict, thresholds: ValidityThresholds) -> dict:
    reasons = []
    recovered = _bool(record.get("object_recovered"))
    if recovered is not True:
        reasons.append("required_object_not_recovered")
    for key, limit in (
        ("translation_cm", thresholds.translation_cm),
        ("rotation_deg", thresholds.rotation_deg),
        ("scale_error_pct", thresholds.scale_error_pct),
        ("penetration_cm", thresholds.penetration_cm),
    ):
        value = _float(record.get(key))
        if value is None:
            reasons.append(f"missing_{key}")
        elif value > limit:
            reasons.append(f"{key}_exceeds_threshold")
    for key in ("support_correct", "collision_valid", "required_visible"):
        value = _bool(record.get(key))
        if value is not True:
            reasons.append(f"{key}_failed" if value is False else f"missing_{key}")
    return {
        **record, "invalid_label": int(bool(reasons)),
        "invalid_reasons": reasons}


def _read(path):
    path = Path(path)
    if path.suffix == ".csv":
        return list(csv.DictReader(path.open()))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else value.get("rows", [])


def generate(input_path, out_dir, thresholds: ValidityThresholds):
    rows = [label_record(row, thresholds) for row in _read(input_path)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{**row, "invalid_reasons": ";".join(row["invalid_reasons"])}
                    for row in rows]
    write_csv(out_dir / "task_validity_labels.csv", serializable)
    write_json(out_dir / "task_validity_labels.json", {
        "thresholds": thresholds.__dict__, "rows": rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--translation-cm", type=float, default=5.0)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--scale-error-pct", type=float, default=15.0)
    parser.add_argument("--penetration-cm", type=float, default=1.0)
    args = parser.parse_args(argv)
    thresholds = ValidityThresholds(
        args.translation_cm, args.rotation_deg,
        args.scale_error_pct, args.penetration_cm)
    rows = generate(args.input, args.out, thresholds)
    print(json.dumps({"n": len(rows), "invalid": sum(r["invalid_label"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
