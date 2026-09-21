"""Validate configs/experiments/icra_contract_v1.yaml (Task 00, plan/00_RESEARCH_CONTRACT.md).

Checks structural completeness of the contract (protocols, gates, table
column -> metric -> protocol mapping) and enforces that
configs/experiments/frozen_fields.yaml has not silently drifted since the
last validated run: a `.frozen_fields.lock.json` sidecar records a content
hash on first run and any later mismatch is a hard error unless
`--bump-frozen-fields` is passed (which requires a --reason).

    python -m agents.eval.validate_contract configs/experiments/icra_contract_v1.yaml --strict
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FIELDS_PATH = ROOT / "configs" / "experiments" / "frozen_fields.yaml"
LOCK_PATH = ROOT / "configs" / "experiments" / ".frozen_fields.lock.json"

REQUIRED_PROTOCOL_KEYS = {"question", "ground_truth", "admissible_baselines", "status"}
REQUIRED_TOP_KEYS = {"version", "frozen_date", "thesis", "scope_decisions", "protocols",
                     "table_column_metric_map", "gates", "stop_condition"}
REQUIRED_GATES = {"G1", "G2", "G3", "G4", "G5", "G6"}


def _canonical_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def check_frozen_fields(bump: bool, reason: str | None) -> list[str]:
    errors = []
    if not FROZEN_FIELDS_PATH.exists():
        errors.append(f"missing {FROZEN_FIELDS_PATH}")
        return errors
    current = yaml.safe_load(FROZEN_FIELDS_PATH.read_text())
    current_hash = _canonical_hash(current)
    if not LOCK_PATH.exists():
        LOCK_PATH.write_text(json.dumps({"hash": current_hash, "reason": "initial freeze"}, indent=2))
        return errors
    lock = json.loads(LOCK_PATH.read_text())
    if lock["hash"] != current_hash:
        if bump and reason:
            LOCK_PATH.write_text(json.dumps({"hash": current_hash, "reason": reason}, indent=2))
        else:
            errors.append(
                "frozen_fields.yaml content changed since last validated run "
                f"(lock hash {lock['hash'][:12]} != current {current_hash[:12]}); "
                "pass --bump-frozen-fields --reason '...' if this is an intentional new version"
            )
    return errors


def validate(contract: dict) -> list[str]:
    errors = []
    missing_top = REQUIRED_TOP_KEYS - contract.keys()
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    protocols = contract.get("protocols", {})
    if not protocols:
        errors.append("no protocols declared")
    for name, spec in protocols.items():
        missing = REQUIRED_PROTOCOL_KEYS - spec.keys()
        if missing:
            errors.append(f"protocol '{name}' missing keys: {sorted(missing)}")

    col_map = contract.get("table_column_metric_map", {})
    for col, protocol in col_map.items():
        if protocol not in protocols:
            errors.append(f"table_column_metric_map['{col}'] references undeclared protocol '{protocol}'")

    gates = contract.get("gates", {})
    missing_gates = REQUIRED_GATES - gates.keys()
    if missing_gates:
        errors.append(f"missing gates: {sorted(missing_gates)}")

    for gid, text in gates.items():
        if not text or not isinstance(text, str):
            errors.append(f"gate {gid} has no textual action")

    if not contract.get("scope_decisions"):
        errors.append("no scope_decisions logged (contract must state what was narrowed and why)")

    if not contract.get("stop_condition"):
        errors.append("no stop_condition declared")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contract_path")
    ap.add_argument("--strict", action="store_true", help="nonzero exit on any error")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bump-frozen-fields", action="store_true")
    ap.add_argument("--reason", default=None)
    args = ap.parse_args()

    contract = yaml.safe_load(Path(args.contract_path).read_text())
    errors = validate(contract)
    errors += check_frozen_fields(args.bump_frozen_fields, args.reason)

    report = {
        "contract_path": args.contract_path,
        "git_sha": _git_sha(),
        "errors": errors,
        "n_protocols": len(contract.get("protocols", {})),
        "n_table_columns_mapped": len(contract.get("table_column_metric_map", {})),
        "pass": len(errors) == 0,
    }

    out_path = Path(args.out) if args.out else ROOT / "validation" / "00" / _git_sha() / "contract_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if errors:
        print(f"\n{len(errors)} error(s) found.")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
