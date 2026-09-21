"""Aggregate a paired harness ledger into the paper's manipulation table."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from robo.eval.harness_spec import load_harness_spec
from robo.eval.harness_validation import VALID_ROLLOUT_OUTCOMES, read_jsonl, validate_records
from robo.eval.metric_utils import paired_bootstrap_delta, write_csv, write_json

STAGES = ("grasp", "lift", "place")
LATEX_NEWLINE = r"\\"


def _task_family(record: dict) -> str:
    value = record.get("task_family")
    if value:
        return str(value)
    task_id = str(record.get("task_id", "")).lower()
    return ("object_to_receptacle" if any(
        token in task_id for token in ("receptacle", "tray", "bowl", "place"))
        else "object_to_region")


def aggregate(records: list[dict], treatment_ids: list[str]) -> tuple[list[dict], dict]:
    by_treatment = defaultdict(list)
    for record in records:
        by_treatment[record.get("treatment_id")].append(record)
    rows, per_reset = [], {}
    for treatment_id in treatment_ids:
        current = by_treatment[treatment_id]
        planned = len(current)
        valid = [r for r in current if r.get("outcome") in VALID_ROLLOUT_OUTCOMES]
        by_family = defaultdict(list)
        for record in current:
            family = _task_family(record)
            if family not in {"object_to_receptacle", "object_to_region"}:
                raise ValueError(f"unknown manipulation task family: {family!r}")
            by_family[family].append(record)
        def success_rate(values):
            return (sum(bool(v.get("success")) for v in values) / len(values)
                    if values else None)
        row = {
            "treatment_id": treatment_id, "planned": planned, "valid": len(valid),
            "coverage": len(valid) / planned if planned else None,
            "object_to_receptacle_planned": len(by_family["object_to_receptacle"]),
            "object_to_region_planned": len(by_family["object_to_region"]),
            "object_to_receptacle": success_rate(by_family["object_to_receptacle"]),
            "object_to_region": success_rate(by_family["object_to_region"]),
            "success": success_rate(current)}
        for stage in STAGES:
            row[stage] = (sum(bool((r.get("stages") or {}).get(stage))
                              for r in current) / planned if planned else None)
        rows.append(row)
        per_reset[treatment_id] = {
            r["reset_state_id"]: float(bool(r.get("success"))) for r in current}
    return rows, per_reset


def render_latex(rows: list[dict], spec) -> str:
    lookup = {row["treatment_id"]: row for row in rows}
    header = (
        r"Treatment axis & Condition & rollout coverage $\uparrow$ & "
        r"obj.$\rightarrow$recept. $\uparrow$ & obj.$\rightarrow$region $\uparrow$ & "
        r"all success $\uparrow$ & grasp $\uparrow$ & lift $\uparrow$ & place $\uparrow$ "
        + LATEX_NEWLINE)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\textbf{Paired harness evaluation across declared intervention axes.} Within each block, the persisted reset-state IDs and complete procedural contract are identical; only the named treatment changes. Rates use the planned-episode denominator for the corresponding task family or full arm, including build, policy, and enhancer failures. A dash denotes a task family with no planned episodes.}",
        r"\label{tab:manipulation}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.3pt}", r"\begin{tabular}{llccccccc}",
        r"\toprule", header, r"\midrule"]
    for comparison_index, comparison in enumerate(spec.comparisons):
        for index, treatment_id in enumerate(comparison.treatments):
            row = lookup[treatment_id]
            axis = (rf"\multirow{{{len(comparison.treatments)}}}{{*}}{{{comparison.axis.title()}}}"
                    if index == 0 else "")
            label = spec.treatments[treatment_id].options.get(
                "label", treatment_id.replace("_", " "))
            values = [row[k] for k in (
                "coverage", "object_to_receptacle", "object_to_region",
                "success", "grasp", "lift", "place")]
            formatted = " & ".join("--" if v is None else f"{100.0 * float(v):.1f}\\%"
                                   for v in values)
            lines.append(f"{axis} & {label} & {formatted} {LATEX_NEWLINE}")
        if comparison_index != len(spec.comparisons) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _table_source(config: dict, ledger_path: str | Path) -> dict:
    directory = Path(__file__).resolve().parent
    digest = lambda data: hashlib.sha256(data).hexdigest()
    return {
        "config_sha256": digest(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()),
        "ledger_sha256": digest(Path(ledger_path).read_bytes()),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip(),
        "implementation_sha256": {name: digest((directory/name).read_bytes()) for name in (
            "main_table.py", "harness_spec.py", "harness_validation.py", "metric_utils.py")},
    }


def generate(config_path: str | Path | dict, ledger_path: str | Path,
             out_dir: str | Path, *, reuse_existing: bool = False) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and not reuse_existing:
        raise ValueError(f"refusing to overwrite manipulation table output: {out_dir}")
    spec = load_harness_spec(config_path)
    records = read_jsonl(ledger_path)
    if spec.raw.get("paper_mode", False) and any(
        r.get("task_family") not in {"object_to_receptacle", "object_to_region"}
        for r in records
    ):
        raise ValueError("paper table requires explicit task_family for every episode")
    declared_ids = spec.raw.get("contract", {}).get("reset_ids")
    if declared_ids is None:
        if spec.raw.get("paper_mode", False):
            raise ValueError("paper table requires predeclared contract.reset_ids")
        planned_ids = {r.get("reset_state_id") for r in records}
        denominator_source = "observed_union_smoke_only"
    else:
        if (not isinstance(declared_ids, list) or not declared_ids
                or any(not isinstance(v, str) or not v for v in declared_ids)
                or len(set(declared_ids)) != len(declared_ids)):
            raise ValueError("contract.reset_ids must be nonempty unique strings")
        planned_ids = set(declared_ids)
        denominator_source = "contract.reset_ids"
    validation = validate_records(records, spec, planned_ids)
    if not validation["ok"]:
        raise ValueError("invalid manipulation table population: " + "; ".join(validation["violations"]))
    rows, per_reset = aggregate(records, list(spec.treatment_ids()))
    comparisons = {}
    for comparison in spec.comparisons:
        baseline = comparison.baseline or comparison.treatments[0]
        comparisons[comparison.id] = {
            treatment: paired_bootstrap_delta(per_reset[baseline], per_reset[treatment])
            for treatment in comparison.treatments if treatment != baseline}
    report = {"rows": rows, "comparisons": comparisons,
              "population_validation": validation, "denominator_source": denominator_source,
              "coverage_definition": "row coverage is valid rollouts / planned episodes; population_validation.coverage is terminal records / planned episodes",
              "degenerate_success_blocks": [
                  {"treatment_id": row["treatment_id"], "planned": row["planned"],
                   "kind": "all_zero" if row["success"] == 0 else "all_one"}
                  for row in rows if row["planned"] and row["success"] in (0, 1)]}
    provenance = {"schema_version": 1, "source": _table_source(spec.raw, ledger_path)}
    members = ("main_table.json", "main_table.csv", "main_table.tex")
    if out_dir.exists():
        expected = {*members, "table_provenance.json"}
        if (out_dir.is_symlink() or {p.name for p in out_dir.iterdir()} != expected
                or any(p.is_symlink() or not p.is_file() for p in out_dir.iterdir())):
            raise ValueError("existing manipulation table is incomplete or has unexpected members")
        saved = json.loads((out_dir/"table_provenance.json").read_text())
        hashes = {name: hashlib.sha256((out_dir/name).read_bytes()).hexdigest() for name in members}
        if saved != {**provenance, "members": hashes}:
            raise ValueError("existing manipulation table source or artifact identity differs")
        if json.loads((out_dir/"main_table.json").read_text()) != report:
            raise ValueError("existing manipulation table differs from current validated ledger")
        return report
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "main_table.json", report)
    write_csv(out_dir / "main_table.csv", rows)
    (out_dir / "main_table.tex").write_text(render_latex(rows, spec))
    provenance["members"] = {
        name: hashlib.sha256((out_dir/name).read_bytes()).hexdigest() for name in members}
    write_json(out_dir / "table_provenance.json", provenance)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.config, args.ledger, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
