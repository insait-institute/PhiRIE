"""Bounded, fail-closed completion trigger for the existing native paper pipeline.

This is scheduling and input freezing only: the canonical ledger, fidelity
producers and scope table helper retain ownership of every measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from robo.eval.native_scale_tables import CORE, _sha, load_units


def read_bound(record):
    path = Path(record["path"])
    if _sha(path) != record["sha256"]:
        raise ValueError(f"release input bytes changed: {path}")
    return json.loads(path.read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path)}


def validate_primary(plan):
    receipt_path = Path(plan["completion_receipt"])
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("state") != "COLLECTED" or receipt.get("collector_returncode") != 0
            or receipt.get("planned") != 2400 or receipt.get("terminal") != 2400
            or receipt.get("missing_units") != 0 or receipt.get("unmeasured_units") != 0
            or receipt.get("all_units_measured") is not True):
        raise ValueError("final primary collector did not establish all2400 measured planned units")
    status = read_bound({"path": receipt["status_path"], "sha256": receipt["status_sha256"]})
    if status["planned"] != 2400 or status["terminal"] != 2400 or any(g["unmeasured"] for g in status["groups"]):
        raise ValueError("final primary status disagrees with completion receipt")
    if _sha(plan["planned_units"]["path"]) != plan["planned_units"]["sha256"]:
        raise ValueError("frozen primary roster changed")
    units = load_units(plan["planned_units"]["path"], plan["final_ledger"])
    if len(units) != 2400 or not all(u["measured"] for u in units):
        raise ValueError("final ledger is incomplete or externally unmeasured")
    populations = {}
    for method in CORE:
        selected = [u for u in units if u["controller_method"] == method]
        ids = {u["canonical_instance_id"] for u in selected}
        if len(selected) != 480 or len(ids) != 48:
            raise ValueError("primary method changes planned population")
        pairs = {(u["canonical_instance_id"], u["reset_id"], u["policy_rng_seed"]) for u in selected}
        if len(pairs) != 480 or any(sum(u["canonical_instance_id"] == iid for u in selected) != 10 for iid in ids):
            raise ValueError("primary method changes reset population")
        populations[method] = pairs
    if any(v != populations[CORE[0]] for v in populations.values()):
        raise ValueError("primary arms do not share the same planned reset population")
    return dict(completion_receipt=bind(receipt_path), status=bind(receipt["status_path"]),
                episode_ledger=bind(plan["final_ledger"]), planned_units=plan["planned_units"],
                planned=2400, measured=2400, executed=sum(u["executed"] for u in units),
                nonexecuted=sum(not u["executed"] for u in units))


def prepare_inputs(plan):
    primary = validate_primary(plan)
    if primary is None:
        return None
    config = read_bound(plan["baseline_config"])
    native = config["native_scale_up"]
    if plan.get("geometry_collection"):
        read_bound(plan["geometry_collection"])
    native.update(planned_units=plan["planned_units"]["path"], episode_ledger=plan["final_ledger"],
                  native_paper_section=False)
    warm_paths = sorted(Path(plan["warm_collections"]).glob("*/collection.json"))
    if not warm_paths:
        raise ValueError("no closed warm appearance boundary")
    warm_path = warm_paths[-1]
    warm = json.loads(warm_path.read_text())
    manifest = read_bound(plan["warm_manifest"])
    if (warm["manifest_sha256"] != plan["warm_manifest"]["sha256"]
            or warm["protocol"] != "capture_train_prelude_v1"
            or warm["planned_render_units"] != len(manifest["rows"]) + 1):
        raise ValueError("warm view/protocol population changed")
    appearance = []
    for unit, record in warm["metrics"].items():
        metric = read_bound(record)
        source = metric["source"]
        ready = warm["ready"].get(unit)
        if (ready is None or source["path"] != ready["path"] or source["sha256"] != ready["sha256"]
                or metric.get("render_protocol") != "capture_train_prelude_v1"):
            raise ValueError("warm metric does not bind its admitted render")
        render = read_bound(ready)
        if render.get("render_protocol") != "capture_train_prelude_v1" or not all(v["byte_exact"] for v in render["native_import_render_identity"]):
            raise ValueError("failed RGB identity cannot enter appearance quality")
        appearance.append(record["path"])
    for records in (warm["failed"], warm["metric_failures"]):
        for record in records.values():
            read_bound(record)
    native["appearance_records"] = appearance
    scope = dict(plan["scope_source"])
    scope_paths = sorted(Path(plan["scope_collections"]).glob("*/collection.json"))
    if not scope_paths:
        raise ValueError("no closed full-roster scope boundary")
    scope_collection = scope_paths[-1]
    closed = json.loads(scope_collection.read_text())
    scope_status = scope_collection.with_name("status.json")
    scope_ledger = scope_status.with_name("episode_ledger.jsonl")
    scope.update(episode_ledger=str(scope_ledger), episode_ledger_sha256=_sha(scope_ledger))
    if (closed["episode_ledger_sha256"] != scope["episode_ledger_sha256"]
            or closed["planned_units_sha256"] != scope["planned_units_sha256"]):
        raise ValueError("closed scope collection does not bind its ledger and roster")
    from robo.eval.native_scale_scope import scope_tables
    scope_output, _ = scope_tables([scope])
    scope_rows = scope_output["T4_scope"] + scope_output["T4_native_controls"]
    status = json.loads(scope_status.read_text())
    if (sum(r["planned"] for r in scope_rows) != 1200 or status["planned"] != 1200
            or sum(g["unmeasured"] for g in status["groups"]) != sum(r["unmeasured"] for r in scope_rows)
            or closed["terminal"] != status["terminal"]
            or closed["executed"] != sum(r["executed"] for r in scope_rows)
            or any(r["scope_run_id"] != closed["scope_run_id"] for r in scope_rows)):
        raise ValueError("scope phase is not the full1200 planned-unit population")
    native["scope_sources"] = [scope]
    boundary = dict(primary=primary, baseline_config=plan["baseline_config"],
        geometry_collection=plan.get("geometry_collection"),
        warm_collection=bind(warm_path), warm_manifest=plan["warm_manifest"],
        warm_render_ready=len(warm["ready"]), warm_metric_ready=len(appearance),
        warm_render_failures=warm["failed"], warm_metric_failures=warm["metric_failures"],
        warm_unattempted=warm["unattempted"], cold_appearance_included=False,
        scope_source=scope, scope_collection=bind(scope_collection), scope_status=bind(scope_status), scope_planned=1200,
        scope_measured=sum(r["measured"] for r in scope_rows),
        scope_unmeasured=sum(r["unmeasured"] for r in scope_rows),
        interpretation="Full primary denominator; separate potentially partial scope and conditional fidelity boundaries. Unavailable primitives and failed render identity stay unmeasured; no scientific promotion.")
    return config, boundary


def run_release(plan, prepared, source_commit, control, *, paper_root=None):
    config, boundary = prepared
    from robo.eval.freeze import reserve_freeze_id
    output_root = Path(plan["output_root"])
    freeze_id = reserve_freeze_id(Path(__file__).resolve().parents[2], output_root)
    stage = output_root / freeze_id
    stage.mkdir()
    work = stage / "sim_recon_sim/scale_up"
    work.mkdir(parents=True)
    boundary_path = work / "input_boundary.json"
    boundary_path.write_text(json.dumps(boundary, indent=2, allow_nan=False) + "\n")
    config["native_scale_up"]["input_boundary"] = bind(boundary_path)
    path = work / "paper_pipeline.json"
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    argv = [sys.executable, "-m", "robo.eval.paper_pipeline", "--config", str(path), "--out", str(work / "paper_tables")]
    if paper_root is not None:
        argv += ["--paper-root", str(paper_root)]
    admission = dict(source_commit=source_commit, source_dirty=False, input_boundary=bind(boundary_path),
                     config=bind(path), command=argv, paper_transfer=paper_root is not None, automatic_claim_promotion=False)
    (work / "pipeline_admission.json").write_text(json.dumps(admission, indent=2) + "\n")
    with (work / "paper_pipeline.log").open("x") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2])
    record = dict(state="GENERATED" if result.returncode == 0 else "PIPELINE_FAILED", returncode=result.returncode,
                  stage=str(stage), command=argv, admission=bind(work / "pipeline_admission.json"),
                  log=bind(work / "paper_pipeline.log"), paper_transfer=paper_root is not None, automatic_claim_promotion=False)
    release = work / "paper_tables/release_manifest.json"
    if result.returncode == 0:
        manifest = json.loads(release.read_text())
        if manifest["planned_units"] != 2400 or manifest["measured_units"] != 2400:
            raise ValueError("generated primary denominator changed")
        for name, digest in manifest["artifacts"].items():
            if _sha(release.parent / name) != digest:
                raise ValueError("generated artifact closure differs")
        record["release_manifest"] = bind(release)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("final table producer requires clean frozen source")
    plan_digest = _sha(args.plan)
    plan = json.loads(args.plan.read_text())
    (args.control / "launch.json").write_text(json.dumps(dict(pid=os.getpid(), source_commit=source, plan=bind(args.plan),
        producer=bind(__file__), max_hours=args.max_hours, automatic_paper_promotion=False), indent=2) + "\n")
    started = time.monotonic()
    record = {"state": "WATCH_LIMIT_REACHED", "paper_transfer": False}
    try:
        while time.monotonic() - started < args.max_hours * 3600:
            if (_sha(args.plan) != plan_digest
                    or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != source
                    or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()):
                raise ValueError("frozen watcher source or plan changed")
            prepared = prepare_inputs(plan)
            if prepared is not None:
                record = run_release(plan, prepared, source, args.control)
                break
            time.sleep(60)
    except Exception as exc:
        record = dict(state="BLOCKED_VALIDATION_OR_ENVIRONMENT", error_type=type(exc).__name__,
                      error=str(exc), paper_transfer=False, automatic_claim_promotion=False)
    (args.control / "completion_receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0 if record["state"] == "GENERATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
