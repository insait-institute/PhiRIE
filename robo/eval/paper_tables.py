"""One provenance-tracked entry point for regenerating paper tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from robo.eval import audit_metrics, fidelity_metrics, main_table, real_world_metrics
from robo.eval.metric_utils import write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(config_path, out_dir, paper_tables_dir=None):
    config_path, out_dir = Path(config_path), Path(out_dir)
    config = yaml.safe_load(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": str(config_path), "config_sha256": sha256(config_path),
              "tables": {}}
    jobs = config.get("tables", {})
    if "manipulation" in jobs:
        job = jobs["manipulation"]
        report["tables"]["manipulation"] = main_table.generate(
            job["harness_config"], job["ledger"], out_dir / "manipulation")
    if "fidelity" in jobs:
        job = jobs["fidelity"]
        report["tables"]["fidelity"] = fidelity_metrics.evaluate_manifest(
            job["manifest"], out_dir / "fidelity",
            lpips_device=job.get("lpips_device", "cpu"))
    if "audit" in jobs:
        job = jobs["audit"]
        report["tables"]["audit"] = audit_metrics.generate(
            job["input"], out_dir / "audit")
    if "real_world" in jobs:
        job = jobs["real_world"]
        report["tables"]["real_world"] = real_world_metrics.generate(
            job["construction"], job.get("trials"), out_dir / "real_world")
    if paper_tables_dir:
        destination = Path(paper_tables_dir)
        destination.mkdir(parents=True, exist_ok=True)
        mapping = {
            out_dir / "manipulation" / "main_table.tex": "frozen_policy_main.tex",
            out_dir / "audit" / "audit_table.tex": "task_local_audit_main.tex"}
        for source, name in mapping.items():
            if source.exists():
                shutil.copy2(source, destination / name)
    write_json(out_dir / "table_provenance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-tables-dir")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.config, args.out, args.paper_tables_dir), indent=2))


if __name__ == "__main__":
    main()
