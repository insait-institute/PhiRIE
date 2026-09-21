"""Canonical end-to-end producer for every quantitative table in the paper.

No numeric cell is edited by hand. This command runs the available aggregators,
archives JSON/CSV/provenance, and optionally copies generated LaTeX files into a
checked-out SimAnyRoom paper repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from robo.eval import (
    audit_loso,
    audit_metrics,
    construction_metrics,
    fidelity_metrics,
    harmony_visual_metrics,
    main_table,
    real_world_table,
)
from robo.eval.metric_utils import write_json

CONSTRUCTION_ROSTER = (
    "GT segments + scan mesh", "Auto discovery + scan mesh",
    "GT segments + splat-fused mesh", "Auto discovery + splat-fused mesh",
    "Single RGB + metric depth",
)
FIDELITY_ROSTER = (
    "Input scene Gaussian, reconstruction ceiling",
    "Factorized composite, GT discovery",
    "Factorized composite, automatic discovery",
    "Composite + Harmonizer Option C",
    "TRELLIS, best single view", "ReconViaGen, multi-view",
    "Evidence-selected proposal", "Evaluation-only oracle candidate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, paper_root: Path | None, destination: str):
    if paper_root is None or not source.exists():
        return None
    target = paper_root / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)



def _checked_audit_artifact(spec: dict, label: str) -> tuple[Path, dict]:
    path = Path(spec["path"]).resolve(strict=True)
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"{label}: source SHA256 mismatch")
    return path, json.loads(path.read_text())


def _fresh_agentic_context(spec: dict, payload: dict) -> dict | None:
    scope = payload.get("study_scope")
    if scope in {None, "conditional_fixed_gt_factory_jobs"}:
        return None
    if (scope != "automatic_training_only_engineering" or payload.get("schema_version") != 1
            or payload.get("headline_eligible") is not False or payload.get("paper_ready") is not False
            or payload.get("retry_claim_status") not in {"case_study_only", "retry_count_threshold_met_preliminary_only"}):
        raise ValueError("unsupported agentic audit schema/scope; no implicit legacy caption")
    if "completion_audit" not in spec:
        raise ValueError("fresh agentic case study requires an authenticated completion audit")
    path, audit = _checked_audit_artifact(spec["completion_audit"], "agentic completion audit")
    full = audit.get("scope") == "complete_cohort_independent_evaluation_integrity"
    if not full and payload.get("retry_claim_status") != "case_study_only":
        raise ValueError("pilot retry scope differs")
    counts = payload.get("counts", {})
    if full and (audit.get("schema_version") != 1 or
            (audit.get("planned_scenes"), audit.get("planned_jobs"), audit.get("terminal_policy_rows")) != (50, 1871, 9355)):
        raise ValueError("full agentic audit requires the complete frozen cohort")
    if (audit.get("status") != "PASS" or audit.get("paper_ready") is not False
            or audit.get("headline_eligible") is not False or audit.get("claim_gate") != "NOT_RUN"
            or audit.get("freeze_id") != payload["freeze_id"]
            or audit.get("construction_freeze_id") != payload.get("construction_freeze_id")
            or audit.get("aggregate_rows_replayed") != payload["rows"]
            or counts.get("scenes") != (audit["planned_scenes"] if full else 1)
            or counts.get("jobs_per_policy") != audit.get("planned_jobs")
            or counts.get("policy_object_rows") != audit.get("terminal_policy_rows")
            or counts.get("policy_object_rows") != 5 * counts.get("jobs_per_policy", 0)
            or counts.get("genuine_retry_jobs") != audit.get("unique_retry_actions")):
        raise ValueError("fresh agentic completion scope/denominator differs")
    required_checks = ("construction_train_only_discovery_binding", "control_frozen_before_evaluation",
        "independent_GT_input_hashes", "matching_and_surface_replay_exact", "unmatched_geometry_null",
        "aggregate_replay_exact", "nonheadline_gates", "retry_new_proposal_transform_and_different_action")
    if full:
        required_checks = tuple(k for k in required_checks if k != "matching_and_surface_replay_exact") + (
            "complete_frozen_roster", "evaluation_metric_shards_authenticated", "runtime_accounting_replay_exact")
        if (type(audit.get("matched_jobs")) is not int or type(audit.get("unmatched_jobs")) is not int
                or min(audit["matched_jobs"], audit["unmatched_jobs"]) < 0
                or audit["matched_jobs"] + audit["unmatched_jobs"] != audit["planned_jobs"]):
            raise ValueError("full agentic matched/unmatched denominator differs")
        runtime_scope = "attributed_initial_generation_plus_canonical_registration_physics_and_retries"
        status = {r["policy_id"]: r.get("runtime_accounting_status") for r in payload["rows"]}
        if (payload.get("runtime_scope") != runtime_scope or audit.get("runtime_scope") != runtime_scope
                or set(status) != {f"A{i}" for i in range(5)} or audit.get("runtime_accounting_status") != status
                or any(v not in {"PASS", "NOT_RUN"} for v in status.values())):
            raise ValueError("full agentic runtime accounting scope/status differs")
        for row in payload["rows"]:
            total = row.get("runtime_minutes_per_scene")
            if row["runtime_accounting_status"] == "NOT_RUN":
                if total is not None:
                    raise ValueError("unmeasured attributed runtime must be null")
            elif any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                    for k in ("runtime_minutes_per_scene", "canonical_runtime_minutes_per_scene", "generation_process_minutes_per_scene")):
                raise ValueError("attributed runtime lacks finite authenticated phase values")
            elif not math.isclose(total, row["canonical_runtime_minutes_per_scene"] +
                    row["generation_process_minutes_per_scene"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("attributed runtime total differs from phase sum")
    if any(audit.get("checks", {}).get(key) != "PASS" for key in required_checks):
        raise ValueError("fresh agentic independence/completion gate differs")
    evidence = audit.get("evidence_hashes", {})
    source = str(Path(spec["path"]).resolve(strict=True))
    if evidence.get(source, {}).get("sha256") != spec["sha256"]:
        raise ValueError("agentic completion audit does not bind the displayed source")
    for member, identity in evidence.items():
        actual = Path(member).resolve(strict=True)
        if _sha256(actual) != identity["sha256"] or actual.stat().st_size != identity["size_bytes"]:
            raise ValueError("agentic completion evidence member changed")
    for row in payload["rows"]:
        geometry, tested, stable = (row.get(key) for key in
            ("geometry_evaluated_jobs", "physical_tested_jobs", "physical_stable_jobs"))
        if (any(type(n) is not int or n < 0 for n in (geometry, tested, stable))
                or geometry > row["accepted_jobs"] or tested > row["accepted_jobs"] or stable > tested
                or geometry > audit.get("matched_jobs", -1)
                or row["planned_jobs"] != counts["jobs_per_policy"]
                or (tested and not math.isclose(row["stable_fraction"], stable / tested))):
            raise ValueError("fresh agentic geometry/physical denominator differs")
    return {"path": str(path), "sha256": spec["completion_audit"]["sha256"],
        "matched_jobs": audit["matched_jobs"], "unmatched_jobs": audit["unmatched_jobs"],
        "source_commit": audit["source_commit"], "evidence_hashes": evidence,
        "full_cohort": full, "scenes": counts["scenes"],
        "runtime_scope_note": (
            "Per-policy attribution of original initial-generation process wall time, including failed and recovery "
            "processes, plus canonical registration, physics and retries. Shared original compute may be charged "
            "to multiple methods and is not fleet elapsed time. Excludes capture, discovery, source Gaussian "
            "training, observation export, evaluation, scheduler queueing and cache materialization. "
            "The completion audit authenticates canonical metric shards and aggregation/runtime replay; it does "
            "not rerun the matcher or provide a second independent geometry estimator." if full else None),
        "runtime_display": "attributed_algorithm_phase" if full else "omitted_incomplete_upstream_generation_accounting"}


def _paired_agentic_source(spec: dict | None, context: dict | None,
                          aggregate: dict) -> tuple[dict, dict] | None:
    """Display only uncertainty already authenticated by the full replay audit."""
    if spec is None:
        return None
    from robo.eval.agentic_ablation import AGENTIC_UNCERTAINTY_PROTOCOL
    if not context or not context["full_cohort"]:
        raise ValueError("paired uncertainty requires a complete cohort audit")
    path, payload = _checked_audit_artifact(spec, "paired uncertainty")
    evidence = context["evidence_hashes"].get(str(path))
    if (not evidence or evidence["sha256"] != spec["sha256"]
            or evidence["size_bytes"] != path.stat().st_size):
        raise ValueError("full replay audit does not authenticate uncertainty")
    if (payload.get("schema_version") != 1
            or payload.get("protocol") != AGENTIC_UNCERTAINTY_PROTOCOL
            or payload.get("freeze_id") != aggregate["freeze_id"]
            or payload.get("construction_freeze_id") != aggregate["construction_freeze_id"]
            or payload.get("planned_scenes") != 50 or payload.get("planned_jobs") != 1871
            or payload.get("paper_ready") is not False
            or payload.get("headline_eligible") is not False or payload.get("claim_gate") != "NOT_RUN"):
        raise ValueError("paired uncertainty source protocol/scope differs")
    expected = [(a, b, metric) for a, b in AGENTIC_UNCERTAINTY_PROTOCOL["contrasts"]
                for metric in AGENTIC_UNCERTAINTY_PROTOCOL["metrics"]]
    rows = payload.get("rows", [])
    if [(r.get("baseline"), r.get("treatment"), r.get("metric")) for r in rows] != expected:
        raise ValueError("paired uncertainty must retain all sixteen predeclared rows")
    by_policy = {r["policy_id"]: r for r in aggregate["rows"]}
    for row in rows:
        a, b, metric = row["baseline"], row["treatment"], row["metric"]
        coverage = metric == "build_coverage"
        geometry = metric in {"f1_20", "cd_cm"}
        support = ("all_planned_jobs" if coverage else "common_accepted_matched_geometry"
                   if geometry else "common_accepted_construction_probes")
        integer_keys = ("planned_scenes", "planned_jobs", "baseline_accepted_jobs", "treatment_accepted_jobs",
                        "baseline_metric_eligible_jobs", "treatment_metric_eligible_jobs", "common_accepted_jobs",
                        "paired_jobs", "paired_scenes", "excluded_pair_jobs", "unsupported_scenes")
        if any(type(row.get(k)) is not int or row[k] < 0 for k in integer_keys):
            raise ValueError("paired uncertainty has invalid integer denominators")
        if (row["contrast"] != f"{b}-{a}" or row["support"] != support
                or row["selection_conditioned"] is not (not coverage)
                or row["independent_physical_validation"] is not False
                or row["planned_jobs"] != 1871 or row["planned_scenes"] != 50
                or row["paired_jobs"] + row["excluded_pair_jobs"] != 1871
                or row["paired_scenes"] + row["unsupported_scenes"] != 50
                or row["paired_scenes"] > row["paired_jobs"]
                or len(row["paired_scene_ids"]) != row["paired_scenes"]
                or row["paired_scene_ids"] != sorted(set(row["paired_scene_ids"]))):
            raise ValueError("paired uncertainty support/counts differ")
        for side, policy in (("baseline", a), ("treatment", b)):
            eligible = 1871 if coverage else by_policy[policy]["geometry_evaluated_jobs" if geometry else "physical_tested_jobs"]
            if (row[f"{side}_accepted_jobs"] != by_policy[policy]["accepted_jobs"]
                    or row[f"{side}_metric_eligible_jobs"] != eligible
                    or row["paired_jobs"] > eligible):
                raise ValueError("paired uncertainty differs from main accepted/eligible denominators")
        if row["common_accepted_jobs"] > min(row["baseline_accepted_jobs"], row["treatment_accepted_jobs"]):
            raise ValueError("paired accepted intersection exceeds marginals")
        if coverage:
            if row["paired_jobs"] != 1871 or row["paired_scenes"] != 50:
                raise ValueError("coverage must retain all planned jobs and scenes")
        elif row["paired_jobs"] > row["common_accepted_jobs"]:
            raise ValueError("conditional pair denominator exceeds accepted intersection")
        values = [row.get(k) for k in ("baseline_mean", "treatment_mean", "delta")]
        if row["paired_jobs"]:
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("paired means/difference must be finite")
            if not math.isclose(values[2], values[1] - values[0], abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError("paired mean difference differs")
            if any(v < 0 or (metric != "cd_cm" and v > 1) for v in values[:2]):
                raise ValueError("paired means violate metric domains")
            if coverage and any(not math.isclose(value, row[f"{side}_accepted_jobs"] / 1871)
                                for side, value in zip(("baseline", "treatment"), values[:2])):
                raise ValueError("paired coverage means differ from planned denominators")
        elif values != [None, None, None]:
            raise ValueError("empty pairs must retain null means/difference")
        ci = row.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError("paired interval requires two endpoints")
        if row["paired_scenes"] < 2:
            if ci != [None, None] or row["status"] != "NOT_ESTIMABLE" or row["reason"] != "fewer_than_two_supported_scenes":
                raise ValueError("insufficient scene support cannot have an estimated interval")
        elif (row["status"] != "ESTIMATED" or row["reason"] is not None
              or any(type(v) not in (int, float) or not math.isfinite(v) for v in ci) or ci[0] > ci[1]):
            raise ValueError("invalid estimated paired interval")
    return payload, {"path": str(path), "sha256": spec["sha256"], "paper_ready": False,
                     "completion_audit": {"path": context["path"], "sha256": context["sha256"]}}


def _fidelity_completion(spec: dict, payload: dict) -> dict | None:
    if "completion_receipt" not in spec:
        return None
    path, receipt = _checked_audit_artifact(spec["completion_receipt"], "fidelity completion receipt")
    table = Path(spec["path"]).resolve(strict=True)
    if (receipt.get("status") != "PASS" or receipt.get("paper_ready") is not False
            or receipt.get("freeze_id", payload["freeze_id"]) != payload["freeze_id"]
            or receipt.get("code_commit") != payload.get("provenance", {}).get("code_commit")
            or Path(receipt["table"]["path"]).resolve(strict=True) != table
            or receipt["table"]["sha256"] != spec["sha256"]):
        raise ValueError("fidelity completion source/producer differs")
    coverage_path, coverage = _checked_audit_artifact(receipt["coverage"], "fidelity coverage")
    populated = [r for r in payload["rows"] if r.get("n_images", 0)]
    if (len(populated) != 1 or populated[0]["method"] != FIDELITY_ROSTER[0]
            or coverage.get("completed_scenes") != coverage.get("planned_scenes")
            or len(receipt["scenes"]) != len(set(receipt["scenes"]))
            or [r["scene_id"] for r in coverage["rows"]] != receipt["scenes"]
            or any(r["status"] != "PASS" for r in coverage["rows"])
            or coverage["planned_scenes"] != populated[0]["n_scenes"]
            or coverage["planned_views"] != populated[0]["n_images"]
            or receipt["n_images"] != populated[0]["n_images"]
            or any(populated[0]["metric_samples"].get(k) != receipt["n_images"] for k in ("psnr","ssim","lpips"))):
        raise ValueError("raw fidelity completion denominator differs")
    manifest = Path(payload["manifest_path"]).resolve(strict=True)
    if _sha256(manifest) != payload["manifest_sha256"]:
        raise ValueError("fidelity metric manifest changed")
    return {"path":str(path),"sha256":spec["completion_receipt"]["sha256"],
        "coverage":{"path":str(coverage_path),"sha256":receipt["coverage"]["sha256"]},
        "metric_manifest":{"path":str(manifest),"sha256":payload["manifest_sha256"]},
        "contract_sha256":receipt["contract_sha256"],"producer_commit":receipt["code_commit"]}

def _droid_cpu_source(path, payload, audit_path, audit):
    """Authenticate the canonical ten-attempt CPU publication without refitting."""
    from robo.manifest.hash import canonical_hash
    from robo.eval.real_world_metrics import summarize_construction
    from robo.eval.real_world_records import validate_alignment

    def checked(identity):
        p = Path(identity["path"])
        if (not p.is_absolute() or p.resolve(strict=True) != p
                or p.stat().st_size != identity["size_bytes"] or _sha256(p) != identity["sha256"]):
            raise ValueError("DROID CPU evidence identity differs")
        return p

    root = path.parent.parent
    if (path != root / "full_cpu_summary/workspaces.json"
            or audit_path != root / "full_cpu_completion_audit.json"
            or payload.get("scope") != "prospective_cpu_alignment" or payload.get("paper_ready") is not False
            or audit.get("schema_version") != 1 or audit.get("scope") != "complete_original_droid_cpu_cohort_integrity"
            or audit.get("status") != "PASS" or audit.get("freeze_id") != root.name
            or audit.get("full_e7_complete") is not False or audit.get("paper_ready") is not False
            or audit.get("planned_workspaces") != 10 or audit.get("result_records") != 10
            or audit.get("not_run_units") != 0):
        raise ValueError("DROID CPU scope, population or completion differs")
    files = audit["artifacts"]
    required = {"workspaces.json", "workspaces.csv", "real_world_table.json",
                "real_world_construction.csv", "real_world_trials.csv"}
    if set(files) != required:
        raise ValueError("DROID CPU canonical publication roster differs")
    for name, identity in files.items():
        if checked(identity) != path.parent / name:
            raise ValueError("DROID CPU publication path differs")
    checked(audit["script"])
    contract_path = root / "contract/freeze_manifest.json"
    if contract_path.resolve(strict=True) != contract_path:
        raise ValueError("DROID CPU contract uses an alias")
    contract = json.loads(contract_path.read_text())
    digest = canonical_hash({k:v for k,v in contract.items() if k not in {"created_utc", "environment", "contract_sha256"}})
    if (digest != contract["contract_sha256"] or digest != audit["contract_sha256"]
            or contract["freeze_id"] != root.name or contract["code"]["dirty"] is not False
            or contract["code"]["commit"] != audit["source_commit"]):
        raise ValueError("DROID CPU original E0/source differs")
    bindings = [r for r in contract["configs"] if r["field"] == "real_world_config"]
    if len(bindings) != 1:
        raise ValueError("DROID CPU original config binding missing")
    binding = bindings[0]
    config_path = Path(binding["resolved_path"])
    if (config_path.resolve(strict=True) != config_path or _sha256(config_path) != binding["source_content_sha256"]):
        raise ValueError("DROID CPU original config changed")
    config = json.loads(config_path.read_text())
    rows = payload["rows"]
    if (len(rows) != 10 or [r["workspace_id"] for r in rows] != [r["workspace_id"] for r in config["captures"]]
            or len({r["workspace_id"] for r in rows}) != 10):
        raise ValueError("DROID CPU attempted workspace roster differs")
    for row, capture in zip(rows, config["captures"]):
        plan = json.loads(checked(capture["plan"]).read_text())
        result_path = checked(row["result_identity"])
        if result_path != root / "real_world/workspaces" / row["workspace_id"] / "result.json":
            raise ValueError("DROID CPU result belongs to another workspace")
        result = json.loads(result_path.read_text())
        evaluation = result["alignment_evaluation"]
        metrics = evaluation["held_out_metrics"] if evaluation else None
        if evaluation:
            for key in ("fit", "reference", "plan"):
                checked(evaluation[key])
        if (row["source"] != "droid" or row["freeze_id"] != root.name or row["source_commit"] != audit["source_commit"]
                or row["capture_id"] != capture["capture_id"] or row["execution_status"] != result["status"]
                or row["execution_status"] not in {"COMPLETE", "FAILED"}
                or row["reconstruction_success"] is not False or row["full_build_success"] is not False
                or row["accepted_objects"] is not None or row["runtime_minutes"] is not None
                or row["cpu_stage_runtime_seconds"] != result["runtime_seconds"]
                or row["failure_reason"] != (result["failure"]["reason"] if result["failure"] else None)
                or row["planned_reference_frames"] != len(plan["split"]["held_out_video_indices"])
                or row["evaluated_reference_frames"] != (evaluation["evaluated_reference_frames"] if evaluation else 0)
                or not 0 <= row["evaluated_reference_frames"] <= row["planned_reference_frames"]
                or row["translation_cm"] != (metrics["center_rms_m"] * 100 if metrics else None)
                or row["rotation_deg"] != (metrics["rotation_residual_deg"]["median"] if metrics else None)
                or row["alignment_pass"] != (evaluation["gate"]["passed"] if metrics else None)):
            raise ValueError("DROID CPU row invents or changes measurement/scope")
        for key in ("translation_cm", "rotation_deg"):
            value = row[key]
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError("DROID CPU invalid alignment value")
        expected = dict(held_out_ids=plan["split"]["held_out_fk_indices"],
                        construction_ids=plan["split"]["train_fk_indices"], selection_ids=plan["split"]["train_fk_indices"],
                        reference_sha256=evaluation["reference"]["sha256"] if metrics else None)
        if row["independent_alignment"] != expected:
            raise ValueError("DROID CPU TRAIN/reference split differs")
        validate_alignment(row)
    if (audit["completed_cpu_units"] != sum(r["execution_status"] == "COMPLETE" for r in rows)
            or audit["failed_cpu_units"] != sum(r["execution_status"] == "FAILED" for r in rows)):
        raise ValueError("DROID CPU terminal denominator differs")
    summary = json.loads((path.parent / "real_world_table.json").read_text())
    if summary != {"construction": summarize_construction(rows), "trials": []}:
        raise ValueError("DROID CPU canonical aggregate differs or physical trials invented")


def _engineering_sources(specs: dict) -> tuple[dict, dict]:
    """Optional sealed engineering evidence; never adds main treatment rows."""
    if not isinstance(specs, dict) or set(specs) - {"trellis2_geometry", "e4_planning", "droid_cpu_alignment", "droid_public_construction", "room_common_view", "task_support_closure", "full_qualification", "full_appearance"}:
        raise ValueError("unknown engineering appendix source")
    if {"room_common_view", "full_appearance"} <= set(specs):
        raise ValueError("select one common-view cohort; do not mix pilot and full appearance")
    if "droid_public_construction" in specs and "droid_cpu_alignment" not in specs:
        raise ValueError("DROID construction stages require the complete CPU cohort")
    payloads, sources = {}, {}
    for name in sorted(specs, key=lambda name: name == "droid_public_construction"):
        spec = specs[name]
        path, payload = _checked_audit_artifact(spec, name)
        if Path(spec["path"]).absolute() != path:
            raise ValueError("engineering source cannot use a symlink or alias")
        audit_path, audit = _checked_audit_artifact(spec["completion_audit"], name + " completion")
        if Path(spec["completion_audit"]["path"]).absolute() != audit_path:
            raise ValueError("engineering completion audit cannot use a symlink or alias")
        context = dict(path=str(path), sha256=spec["sha256"], paper_ready=False,
                       completion_audit=dict(path=str(audit_path), sha256=spec["completion_audit"]["sha256"]))
        if name == "full_qualification":
            from robo.eval.paper_full_qualification import validate_source
            context.update(validate_source(spec, path, payload, audit_path, audit))
        elif name == "task_support_closure":
            from robo.eval.paper_task_support_closure import validate_closure
            if path != audit_path or payload != audit:
                raise ValueError("task-support closure must use the original complete QA source")
            verified = validate_closure(path, expected_sha256=spec["sha256"])
            # Keep the original QA fields, so each displayed count has an exact
            # source-field pointer rather than a synthetic normalized row.
            context.update(producer_commit=verified["producer_commit"],
                contract_sha256=verified["contract_sha256"],
                feature_contract_sha256=verified["feature_contract_sha256"],
                feature_seal=verified["feature_seal"],
                label_join_manifest=verified["label_join_manifest"])
        elif name in {"room_common_view", "full_appearance"}:
            if name == "full_appearance":
                from robo.eval.paper_full_appearance import validate_source
            else:
                from robo.eval.paper_room_diagnostic import validate_source
            payload, coverage, extra = validate_source(spec, path, payload, audit_path, audit)
            context.update(extra)
            coverage_path = audit_path.parent / "coverage.json"
            payloads[name + "_coverage"] = coverage
            sources[name + "_coverage"] = dict(path=str(coverage_path),
                sha256=_sha256(coverage_path), paper_ready=False,
                completion_audit=context["completion_audit"])
        elif name == "droid_public_construction":
            from robo.eval.paper_droid_stages import validate_stages
            if path != audit_path or payload != audit:
                raise ValueError("DROID public construction must use the full completion source")
            gs_path, gs = _checked_audit_artifact(spec["gaussian_completion_audit"], "DROID Gaussian completion")
            if Path(spec["gaussian_completion_audit"]["path"]).absolute() != gs_path:
                raise ValueError("DROID Gaussian completion cannot use an alias")
            payload = validate_stages(path, payload, gs_path, gs,
                [r["workspace_id"] for r in payloads["droid_cpu_alignment"]["rows"]],
                dict(stage_root=str(Path(sources["droid_cpu_alignment"]["path"]).parent.parent),
                     commit=sources["droid_cpu_alignment"]["producer_commit"]))
            context.update(contract_sha256=audit["contract_sha256"], producer_commit=audit["source_commit"],
                gaussian_completion_audit=dict(path=str(gs_path),sha256=spec["gaussian_completion_audit"]["sha256"]),
                gaussian_contract_sha256=gs["contract_sha256"], gaussian_producer_commit=gs["source_commit"])
        elif name == "droid_cpu_alignment":
            _droid_cpu_source(path, payload, audit_path, audit)
            context["contract_sha256"] = audit["contract_sha256"]
            context["producer_commit"] = audit["source_commit"]
            context["artifacts"] = audit["artifacts"]
        elif name == "trellis2_geometry":
            required = {"source_train_registration_before_GT", "original_source_validators", "complete15_roster",
                        "three_frozen_matches", "canonical_geometry_replay_exact", "aggregate_arithmetic_exact",
                        "CSV_JSON_exact", "unmatched_null_geometry", "appearance_null", "sealed_outputs", "nonheadline_gates"}
            if (payload.get("scope") != "trellis2_mesh_only_independent_geometry_engineering_pilot"
                    or payload.get("schema_version") != 1 or audit.get("schema_version") != 1
                    or audit.get("scope") != "trellis2_mesh_geometry_output_integrity" or audit.get("status") != "PASS"
                    or audit.get("source_commit") != payload.get("source_commit")
                    or audit.get("freeze_id") != payload.get("freeze_id")
                    or any(payload.get(k) is not False or audit.get(k) is not False for k in
                           ("native_gaussian", "full_twin_ready", "paper_ready", "headline_eligible"))
                    or payload.get("claim_gate") != "NOT_RUN" or audit.get("claim_gate") != "NOT_RUN"
                    or not required.issubset(audit.get("checks", {}))
                    or any(audit["checks"].get(k) != "PASS" for k in required)):
                raise ValueError("T2 engineering scope/audit/claim gates differ")
            if (path.name != "geometry.json" or path.parent.name != "trellis2_geometry"
                    or audit_path != path.parent.parent / "independent_geometry_completion_audit.json"
                    or payload.get("planned_jobs") != 15 or payload.get("generated_jobs") != 15 or payload.get("geometry_evaluated_jobs") != 3
                    or payload.get("unmatched_jobs") != 12 or len(payload.get("rows", [])) != 15):
                raise ValueError("T2 engineering denominator or output path differs")
            evidence = audit.get("evidence_hashes", {})
            if str(path) not in evidence:
                raise ValueError("T2 source missing from completion evidence")
            for file, record in evidence.items():
                f = Path(file)
                if (str(f.resolve(strict=True)) != file or _sha256(f) != record["sha256"]
                        or f.stat().st_size != record.get("bytes", record.get("size_bytes"))):
                    raise ValueError("T2 completion evidence bytes changed")
            if {r["job_id"] for r in payload["rows"]} != {f"38d58a7a31/obj_{i}" for i in range(1000,1015)}:
                raise ValueError("T2 engineering planned roster differs")
            matched = [r for r in payload["rows"] if r["geometry_status"] == "matched"]
            if len(matched) != 3:
                raise ValueError("T2 matched denominator differs")
            for row in payload["rows"]:
                if (any(row.get(k) is not None for k in ("psnr", "ssim", "lpips"))
                        or row["geometry_status"] not in {"matched", "unmatched"}
                        or (row["geometry_status"] == "unmatched" and
                            any(row.get(k) is not None for k in ("cd_cm", "f1_20", "collapse", "matched_gt_id")))):
                    raise ValueError("T2 unavailable appearance or geometry invented")
            for key in ("cd_cm", "f1_20"):
                values = [r[key] for r in matched]
                if (any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values)
                        or not math.isclose(payload[key], sum(values)/len(values), rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError("T2 aggregate arithmetic differs")
            if any(r["f1_20"] > 1 or type(r["collapse"]) is not bool or r["collapse"] != (r["f1_20"] < .1) for r in matched):
                raise ValueError("T2 canonical F1/collapse definition differs")
            if payload["catastrophic_collapses"] != sum(r["collapse"] for r in matched):
                raise ValueError("T2 collapse count differs")
        else:
            receipt_path, receipt = _checked_audit_artifact(spec["publication_receipt"], "E4 publication")
            root = path.parent.parent
            if Path(spec["publication_receipt"]["path"]).absolute() != receipt_path:
                raise ValueError("E4 publication receipt cannot use a symlink or alias")
            if (path.name != "handoff.json" or path.parent.name != "planned_coverage"
                    or receipt_path != root / "publication_receipt.json" or audit_path != root / "independent_qa.json"
                    or payload.get("scope") != "compact_harness_planning_unqualified_coverage"
                    or payload.get("paper_ready") is not False or payload.get("policy_launch_allowed") is not False
                    or payload.get("execution_status") != "NOT_RUN" or payload.get("applicability_gate") != "FAIL"
                    or payload.get("rollout_ledger") is not None or payload.get("actual_reset_bank") is not None
                    or payload.get("planned_episode_cells") != 40 or payload.get("planned_semantic_pairs") != 28
                    or payload.get("planned_qualification_cells") != 280 or payload.get("recorded_rollout_episodes") != 0
                    or receipt.get("freeze_id") != root.name
                    or payload.get("freeze_id", root.name) != root.name
                    or receipt.get("source_commit") != payload.get("source_commit")
                    or receipt.get("planned_cells") != 40 or receipt.get("executed_policy_cells") != 0
                    or receipt.get("applicability_gate") != "FAIL" or receipt.get("manipulation_claim_gate") != "NOT_RUN"
                    or audit.get("integrity") != "PASS" or audit.get("planned_episode_cells") != 40
                    or audit.get("policy_executed_cells") != 0 or audit.get("manipulation_claim_gate") != "NOT_RUN"):
                raise ValueError("E4 planning-only scope/gates/denominators differ")
            manifest_path = path.parent / "manifest.json"
            if manifest_path.absolute() != manifest_path.resolve() or (path.parent / "seal.json").absolute() != (path.parent / "seal.json").resolve():
                raise ValueError("E4 manifest/seal uses a symlink")
            manifest = json.loads(manifest_path.read_text())
            seal_path = path.parent / "seal.json"
            seal = json.loads(seal_path.read_text())
            if (_sha256(manifest_path) != receipt["coverage_manifest_sha256"]
                    or _sha256(manifest_path) != audit["manifest_sha256"]
                    or _sha256(seal_path) != receipt["coverage_seal_sha256"]
                    or seal["members"]["manifest.json"]["sha256"] != receipt["coverage_manifest_sha256"]
                    or set(manifest["files"]) != {"handoff.json", "planned_episode_cells.json", "planned_reset_definitions.json"}):
                raise ValueError("E4 coverage seal/manifest differs")
            if (manifest.get("freeze_id", root.name) != root.name
                    or manifest.get("planning_terminal") != payload.get("planning_terminal")
                    or Path(payload["planning_terminal"]["path"]) != root / "planning_terminal.json"):
                raise ValueError("E4 manifest/handoff must bind the same stage terminal")
            for file, record in manifest["files"].items():
                f = path.parent / file
                if f.absolute() != f.resolve() or _sha256(f) != record["sha256"] or f.stat().st_size != record["size_bytes"]:
                    raise ValueError("E4 sealed coverage member changed")
            terminal_path, terminal = _checked_audit_artifact(payload["planning_terminal"], "E4 terminal")
            if _sha256(terminal_path) != receipt["terminal_sha256"] or terminal.get("applicability_status") != "FAIL":
                raise ValueError("E4 planning terminal differs")
            cells_payload = json.loads((path.parent / "planned_episode_cells.json").read_text())
            cells = cells_payload
            if len(cells) != 40 or len({r["episode_id"] for r in cells}) != 40:
                raise ValueError("E4 planned cell roster differs")
            from run.icra2027.e4_compact_harness import checked_protocol, definitions
            protocol = checked_protocol(payload["protocol"]["path"], payload["protocol"]["sha256"])
            expected = {(r.scene_id, r.task_id, r.reset_state_id, r.reset_seed, arm)
                        for r in definitions(protocol) for arm in ("A0", "A4")}
            if {(r["scene_id"],r["task_id"],r["reset_state_id"],r["reset_seed"],r["construction_policy"]) for r in cells} != expected:
                raise ValueError("E4 canonical fixed task/reset/treatment roster differs")
            null_fields = {"outcome", "success", "score", "grasp", "lift", "place", "ticks", "task_definition",
                           "reset_provenance", "camera_diagnostics", "policy_latency_ms", "observation_latency_ms"}
            if set(audit["null_measurement_fields"]) != null_fields:
                raise ValueError("E4 measurement null-field contract differs")
            for row in cells:
                if (row.get("execution_status") != "NOT_RUN" or row.get("policy_execution") != "not_invoked_prebuild"
                        or not null_fields.issubset(row) or any(row[k] is not None for k in null_fields)):
                    raise ValueError("E4 invented policy or reset/camera telemetry")
            summaries = payload["source_qualification_summaries"]
            if (summaries != receipt["source_qualification_summaries"] or summaries.get("27dd4da69e") is not None
                    or set(summaries) != {"27dd4da69e", "40aec5fffa"}
                    or {k:summaries["40aec5fffa"][k] for k in ("planned_cells","passed_cells","fixed_selected_cells","fixed_passed_cells")}
                       != dict(planned_cells=100,passed_cells=0,fixed_selected_cells=20,fixed_passed_cells=0)):
                raise ValueError("E4 observed qualifier summary differs")
            for group in summaries.values():
                if group is None: continue
                for key in ("gate", "manifest", "metrics", "seal"):
                    f = Path(payload["planning_applicability_source"]["e0"]["path"]).parents[4] / group[key]["path"]
                    if _sha256(f) != group[key]["sha256"] or f.stat().st_size != group[key]["size_bytes"]:
                        raise ValueError("E4 original qualifier summary evidence changed")
            from robo.manifest.hash import canonical_hash
            contract_path = root / "contract/freeze_manifest.json"
            contract = json.loads(contract_path.read_text())
            digest = canonical_hash({k:v for k,v in contract.items() if k not in {"created_utc","environment","contract_sha256"}})
            if (contract.get("freeze_id") != root.name
                    or digest != contract.get("contract_sha256") or digest != receipt["contract_sha256"]
                    or contract["code"].get("commit") != payload["source_commit"] or contract["code"].get("dirty") is not False
                    or manifest["code"].get("commit") != payload["source_commit"]):
                raise ValueError("E4 coverage E0/source closure differs")
            context["publication_receipt"] = dict(path=str(receipt_path),sha256=spec["publication_receipt"]["sha256"])
        payloads[name], sources[name] = payload, context
    return payloads, sources


def _audited_tables(config: dict, config_path: Path, out_dir: Path,
                    paper_root: Path | None) -> dict:
    """Format existing producer outputs without recomputing/promoting metrics.

    This deliberately supports a preliminary audit only. Hash identity proves
    which archived values are displayed, not the missing generation provenance.
    """
    if config.get("mode") != "preliminary_audit":
        raise ValueError("audited_artifacts require mode=preliminary_audit")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite paper audit: {out_dir}")
    artifacts = config["audited_artifacts"]
    if set(artifacts) != {"construction", "fidelity", "agentic"}:
        raise ValueError("audit requires construction, fidelity, and agentic sources")
    payloads, sources = {}, {}
    for name, spec in artifacts.items():
        source = Path(spec["path"]).resolve(strict=True)
        observed = _sha256(source)
        if observed != spec["sha256"]:
            raise ValueError(f"{name}: source SHA256 mismatch")
        payloads[name] = json.loads(source.read_text())
        sources[name] = {"path": str(source), "sha256": observed,
                         "paper_ready": payloads[name].get("paper_ready", False)}
    agentic_context = _fresh_agentic_context(artifacts["agentic"], payloads["agentic"])
    fidelity_context = _fidelity_completion(artifacts["fidelity"], payloads["fidelity"])
    if agentic_context: sources["agentic"]["completion_audit"] = agentic_context
    if fidelity_context: sources["fidelity"]["completion_receipt"] = fidelity_context
    e3_rows = payloads["agentic"]["rows"]
    if [r["policy_id"] for r in e3_rows] != ["A0", "A1", "A2", "A3", "A4"]:
        raise ValueError("agentic policy roster must be exactly A0--A4")
    planned = {r["planned_jobs"] for r in e3_rows}
    if len(planned) != 1 or next(iter(planned)) <= 0:
        raise ValueError("agentic planned denominators differ or are empty")
    for row in e3_rows:
        if not 0 <= row["accepted_jobs"] <= row["planned_jobs"]:
            raise ValueError("invalid accepted/planned count")
        if not math.isclose(row["build_coverage"],
                            row["accepted_jobs"] / row["planned_jobs"]):
            raise ValueError("agentic coverage does not match denominator")
    construction_names = [r["regime"] for r in payloads["construction"]["rows"]]
    if sorted(construction_names) != sorted(CONSTRUCTION_ROSTER):
        raise ValueError("construction requires exact five planned regime names")
    construction_context = None
    if "completion_audit" in artifacts["construction"]:
        from robo.eval.paper_full_construction import validate_source
        spec = artifacts["construction"]
        audit_path, audit = _checked_audit_artifact(spec["completion_audit"], "construction completion")
        construction_context = validate_source(spec, Path(sources["construction"]["path"]),
                                               payloads["construction"], audit_path, audit)
        sources["construction"]["completion_audit"] = construction_context
    fidelity = payloads["fidelity"]
    if fidelity.get("validation", {}).get("valid") is not True:
        raise ValueError("fidelity source did not pass validation")
    fidelity_names = [r["method"] for r in fidelity["rows"]]
    if sorted(fidelity_names) != sorted(FIDELITY_ROSTER):
        raise ValueError("fidelity requires exact eight method names, including missing")
    for row in fidelity["rows"]:
        expected_unit = "room" if row["method"] in FIDELITY_ROSTER[:4] else "object"
        if row["unit"] != expected_unit:
            raise ValueError(f"wrong fidelity unit for {row['method']}")

    engineering_payloads, engineering_sources = _engineering_sources(config.get("engineering_appendix", {}))
    payloads.update(engineering_payloads); sources.update(engineering_sources)
    paired = _paired_agentic_source(config.get("agentic_paired_uncertainty"), agentic_context, payloads["agentic"])
    if paired:
        payloads["agentic_paired_uncertainty"], sources["agentic_paired_uncertainty"] = paired
    make_paired_figure = config.get("agentic_paired_uncertainty_figure", False)
    if type(make_paired_figure) is not bool or (make_paired_figure and not paired):
        raise ValueError("paired figure requires a boolean option and authenticated full-cohort statistics")
    concise = config.get("concise_captions", False)
    if type(concise) is not bool or (concise and not (agentic_context and agentic_context["full_cohort"])):
        raise ValueError("concise captions require an authenticated full controller cohort")
    claims = []
    def fmt(name, index, key, *, percent=False, digits=3, element=None):
        if index is None:
            value = payloads[name]
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
        else:
            value = payloads[name]["rows"][index].get(key)
        if element is not None:
            value = value[element]
        field = key if index is None else f"rows[{index}].{key}"
        if element is not None:
            field += f"[{element}]"
        if value is not None and (not isinstance(value, (int, float)) or
                                  not math.isfinite(value)):
            raise ValueError(f"{name}.rows[{index}].{key} is not finite")
        rendered = "--" if value is None else (
            f"{value * 100:.1f}\\%" if percent else f"{value:.{digits}f}")
        claims.append({"claim_id": f"{name}.{index}.{key}" + (f"[{element}]" if element is not None else ""), "paper_file": name,
                       "source_artifact": sources[name]["path"],
                       "source_sha256": sources[name]["sha256"],
                       "source_field": field,
                       "value_in_source": value, "value_in_text": rendered,
                       "status": "unavailable" if value is None else "rounded",
                       "claim_gate": "NOT_RUN", "owner": "E9"})
        return rendered

    def table(caption, label, columns, header, rows, *, wide=False):
        concise_text = {
            "tab:agentic": "Complete automatic-discovery controller cohort. Coverage retains all planned jobs; geometry uses matched accepted jobs. "
                "A4 changes that subset. Stable/tested is the acceptance-time construction probe. Attributed runtime includes initial generation, "
                "canonical registration, probes and retries, including failed attempts; it is not fleet or capture-to-simulator time.",
            "tab:fidelity": "Held-out raw input-Gaussian appearance with the complete view roster. Other method rows remain unmeasured.",
            "tab:room-common-view": "Common-camera appearance diagnostic. Both rows use the same predeclared TEST views; paired/planned retains "
                "the original two-scene denominator. Only one composite scene is available; no population effect is inferred.",
            "tab:trellis2-engineering": "Separate TRELLIS.2 mesh pilot. Independent geometry is conditional on fixed matched objects; "
                "unmatched jobs remain null. No appearance, native Gaussian or manipulation result is implied.",
            "tab:compact-applicability": "Fixed compact manipulation applicability. Checked cells are prerequisite tests, not policy attempts. "
                "No rollout or success rate is measured; all planned cells remain included.",
            "tab:task-support-closure": "Task-support applicability under fixed evidence requirements. Every query lacks required robot/camera "
                "support; all-invalid folds prevent LOSO estimation. Predictive metrics remain unmeasured, not zero.",
            "tab:droid-cpu-alignment": "Prospective DROID stages. Eval./planned precedes held-out position RMS (cm) and median orientation residual "
                "(deg). GS: completed (C) or unavailable (--); prep.: prepared proposals, not accepted simulator assets. "
                "All workspaces and failures are retained.",
        }
        if concise:
            caption = concise_text.get(label, caption)
        env = "table*" if wide else "table"
        width = r"\textwidth" if wide else r"\columnwidth"
        return "\n".join([
            f"% Generated by robo.eval.paper_pipeline; audit {config['freeze_id']}",
            f"\\begin{{{env}}}[t]", r"\centering\scriptsize",
            r"\caption{" + ("" if concise else r"\textbf{Preliminary audit.} ") + caption + "}",
            f"\\label{{{label}}}", r"\setlength{\tabcolsep}{3pt}",
            (f"\\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep{{\\fill}}}}{columns}@{{}}}}"
             if wide else f"\\resizebox{{{width}}}{{!}}{{%\n\\begin{{tabular}}{{{columns}}}"),
            r"\toprule", header + r" \\", r"\midrule", *rows,
            r"\bottomrule", (r"\end{tabular*}" if wide else r"\end{tabular}}"),
            f"\\end{{{env}}}", ""])

    construction_rows = []
    for i in sorted(range(len(construction_names)),
                    key=lambda i: CONSTRUCTION_ROSTER.index(construction_names[i])):
        row = payloads["construction"]["rows"][i]
        cells = [row["regime"], fmt("construction", i, "instances", digits=0) +
                 "/" + fmt("construction", i, "planned_scenes", digits=0),
                 fmt("construction", i, "accepted_instances", digits=0),
                 fmt("construction", i, "yield", percent=True),
                 fmt("construction", i, "f1_20"),
                 (fmt("construction", i, "stable_instances", digits=0) + "/" +
                  fmt("construction", i, "tested_instances", digits=0) if construction_context else
                  fmt("construction", i, "stability", percent=True)),
                 fmt("construction", i, "runtime_minutes", digits=1)]
        construction_rows.append(" & ".join(cells) + r" \\")
    generated = {"automatic_construction_main.tex": table(
        "legacy construction inventory. Counts retain every planned scene. "
        "Geometry values are historical diagnostics, not independent held-out "
        "scores; invalid geometry is withheld (--). Original build provenance "
        "is incomplete for every regime.", "tab:construction", "lcccccc",
        r"Regime & input/scenes & accepted & yield & F1@20 & stable & min/scene",
        construction_rows)}
    if construction_context:
        generated["automatic_construction_main.tex"] = table(
            "Full planned input ladder, with one measured export/drop stage. Export yield retains all discovered inputs; "
            "stable/tested uses an independent isolated-body drop protocol, distinct from the controller's acceptance probe "
            "and full-room task qualification. Other input families, exported-body held-out geometry and complete runtime remain "
            "unmeasured. Successful exports do not establish complete interactive simulators or manipulation.",
            "tab:construction", "lcccccc",
            r"Regime & input/scenes & exports & yield & F1@20 & stable/tested & min/scene",
            construction_rows, wide=True)
    room, objects = [], []
    for i in sorted(range(len(fidelity_names)),
                    key=lambda i: FIDELITY_ROSTER.index(fidelity_names[i])):
        row = fidelity["rows"][i]
        if row["unit"] == "room":
            room.append(" & ".join([row["method"],
                fmt("fidelity", i, "n_scenes", digits=0),
                fmt("fidelity", i, "n_images", digits=0),
                fmt("fidelity", i, "psnr", digits=2),
                fmt("fidelity", i, "ssim"), fmt("fidelity", i, "lpips")]) + r" \\")
        else:
            objects.append(" & ".join([row["method"],
                fmt("fidelity", i, "n_objects", digits=0),
                fmt("fidelity", i, "psnr", digits=2), fmt("fidelity", i, "ssim"),
                fmt("fidelity", i, "cd_cm"), fmt("fidelity", i, "f1_20"),
                fmt("fidelity", i, "collapses", digits=0)]) + r" \\")
    room_caption = ("held-out raw-room appearance from fresh TRAIN-only Gaussians. Only the input "
        "Gaussian row is populated; factorized composites and Harmonizer remain unmeasured. "
        "The full method matrix remains incomplete." if fidelity_context else
        "held-out room appearance. All populated conditions use the same official "
        "test views; Harmonizer has no evaluated frames. The full method matrix remains incomplete.")
    generated["object_factorization_main.tex"] = table(
        room_caption, "tab:fidelity", "lccccc",
        "Representation & scenes & views & PSNR & SSIM & LPIPS", room, wide=True)
    omit_objects = config.get("omit_unmeasured_object_table", False)
    if type(omit_objects) is not bool:
        raise ValueError("omit_unmeasured_object_table must be boolean")
    if omit_objects and any(row.get(key) is not None
            for row in fidelity["rows"] if row["unit"] == "object"
            for key in ("psnr", "ssim", "lpips", "cd_cm", "f1_20", "collapses", "precision20", "recall20")):
        raise ValueError("cannot omit an object table containing measured metrics")
    object_table = table(
        ("object fidelity availability. The declared object-method comparison remains unmeasured; " +
         ("the separate controller cohort reports geometry only for matched jobs. " if agentic_context and agentic_context["full_cohort"] else
          "the separate controller case study reports geometry only for matched jobs. ") +
         "Dashes are unmeasured, not zero." if agentic_context else
         "object fidelity availability. No independent masked-view/surface "
         "evaluation is available; dashes are unmeasured, not zero."),
        "tab:object-fidelity", "lcccccc",
        "Proposal & objects & mPSNR & mSSIM & CD [cm] & F1@20 & collapses",
        objects, wide=True)
    if omit_objects:
        generated["object_fidelity_unmeasured_skeleton.tex"] = object_table
    else:
        generated["object_factorization_main.tex"] += object_table
    labels = ["A0: fixed TRELLIS", "A1: fixed priority", "A2: evidence selection",
              "A3: bounded registration retry", "A4: retry + abstention"]
    rows = []
    for i, label in enumerate(labels):
        cells = [label, fmt("agentic", i, "accepted_jobs", digits=0) + "/" +
                 fmt("agentic", i, "planned_jobs", digits=0),
                 fmt("agentic", i, "build_coverage", percent=True)]
        if agentic_context:
            cells.append(fmt("agentic", i, "geometry_evaluated_jobs", digits=0))
        cells += [fmt("agentic", i, "f1_20"), fmt("agentic", i, "cd_cm"),
                  fmt("agentic", i, "catastrophic_collapses", digits=0)]
        cells.append((fmt("agentic", i, "physical_stable_jobs", digits=0) + "/" +
                      fmt("agentic", i, "physical_tested_jobs", digits=0)) if agentic_context else
                     fmt("agentic", i, "stable_fraction", percent=True))
        cells += [fmt("agentic", i, "retry_count", digits=0), fmt("agentic", i, "abstain_count", digits=0)]
        if not agentic_context or agentic_context["full_cohort"]:
            cells.append(fmt("agentic", i, "runtime_minutes_per_scene", digits=2))
        rows.append(" & ".join(cells) + r" \\")
    caption = (
        "automatic-discovery controller case study in one predeclared scene. Initial proposals "
        "and planned jobs are shared; construction is isolated from evaluation GT. Geometry "
        "is conditional on matched jobs (geom. n), with unmatched jobs retained in coverage. "
        "A4 has a different accepted geometry subset. Stable/tested counts describe the same "
        "construction probe used for acceptance, not independent manipulation. A3/A4 share "
        "the same registration retries. Runtime is omitted because this archived aggregate "
        "excludes upstream generation and other stages; no full-cohort headline claim is enabled."
        if agentic_context else
        "conditional controller ablation. Initial proposals and planned jobs "
        "are shared. A4 quality and stability are conditional on acceptance. "
        "Retry is a signed-axis registration action. GT-assisted legacy crops "
        "and incomplete generator provenance prevent a headline claim. "
        "Stability is a construction probe, not independent manipulation.")
    if agentic_context and agentic_context["full_cohort"]:
        caption = (f"automatic-discovery controller audit across the complete declared {agentic_context['scenes']}-scene engineering cohort. "
            "Coverage retains every planned job; geometry is conditional on matched accepted jobs, with a different A4 subset. "
            "Stable/tested counts are construction evidence, not independent manipulation. A3/A4 share registration retries. "
            "Attributed min/scene includes original generation processes, failures and recoveries, plus canonical registration, "
            "physics and retries. This is neither capture-to-sim nor fleet elapsed time. Dashes denote incomplete timing evidence. "
            "These construction results do not establish manipulation gains.")
    header = ("Policy & accepted/planned & coverage & geom. n & F1@20 & CD [cm] & collapses & stable/tested & retries & abstain"
              if agentic_context else
              "Policy & accepted/planned & coverage & F1@20 & CD [cm] & collapses & stable & retries & abstain & min/scene")
    if agentic_context and agentic_context["full_cohort"]:
        header += " & attrib. min/scene"
    alignment = "lcccccccccc" if agentic_context and agentic_context["full_cohort"] else "lccccccccc"
    generated["agentic_ablation_main.tex"] = table(caption, "tab:agentic", alignment, header, rows, wide=True)

    if paired:
        name = "agentic_paired_uncertainty"
        metric_labels = {"build_coverage": "Coverage", "f1_20": "F1@20", "cd_cm": "CD [cm]",
                         "stable_fraction": "Stable fraction (probe)"}
        paired_rows = []
        for i, row in enumerate(payloads[name]["rows"]):
            cells = [row["contrast"], metric_labels[row["metric"]],
                     fmt(name, i, "paired_jobs", digits=0) + "/" + fmt(name, i, "planned_jobs", digits=0),
                     fmt(name, i, "paired_scenes", digits=0), fmt(name, i, "delta"),
                     "[" + fmt(name, i, "ci95", element=0) + ", " + fmt(name, i, "ci95", element=1) + "]"]
            paired_rows.append(" & ".join(cells) + r" \\")
        generated["agentic_paired_uncertainty.tex"] = table(
            "fixed treatment-minus-baseline contrasts. Coverage includes every planned object; geometry uses only "
            "common accepted, matched objects and stability uses common accepted construction probes. "
            "Whole supported scenes are resampled; intervals are pointwise descriptive, not multiplicity-adjusted "
            "or seed-replicated. Fewer than two supported scenes gives no interval (--). Conditional differences "
            "do not establish population-wide quality or independent physics gains. Fraction differences use units of one.",
            "tab:agentic-paired", "llcccc", "Contrast & Metric & paired/planned & scenes & difference & 95\\% interval",
            paired_rows, wide=True)

    if "trellis2_geometry" in engineering_payloads:
        name = "trellis2_geometry"
        cells = ["TRELLIS.2 mesh", *[fmt(name, None, key, digits=0) for key in
                 ("planned_jobs", "geometry_evaluated_jobs", "unmatched_jobs")],
                 fmt(name, None, "cd_cm"), fmt(name, None, "f1_20"),
                 fmt(name, None, "catastrophic_collapses", digits=0)]
        generated["trellis2_geometry_engineering.tex"] = table(
            "separate mesh-only engineering pilot. Geometry is conditional on the fixed matched objects; unmatched jobs retain null geometry. "
            "Recorded TRAIN registration and independent references are reused unchanged. No appearance, native Gaussian, full-twin or manipulation result is implied.",
            "tab:trellis2-engineering", "lcccccc", "Generator & Planned & Matched & Unmatched & CD (cm) & F1@20 & Collapses",
            [" & ".join(cells) + r" \\"])
    if "e4_planning" in engineering_payloads:
        name = "e4_planning"
        keys = ("planned_episode_cells", "source_qualification_summaries.40aec5fffa.fixed_selected_cells",
                "source_qualification_summaries.40aec5fffa.fixed_passed_cells", "recorded_rollout_episodes")
        cells = ["Fixed compact matrix", *[fmt(name, None, key, digits=0) for key in keys], "--"]
        generated["compact_applicability_engineering.tex"] = table(
            "planning applicability for fixed policy cells. Prerequisite records include unavailable endpoints and construction failures; "
            "One scene has no size-admissible common target; "
            "the other scene's checked cells fail construction prerequisites. No simulator or policy rollout is executed. "
            "The remaining planned cells are unexecuted, not measured policy failures.",
            "tab:compact-applicability", "lccccc", "Protocol & Planned & Prereq. checked & Eligible & Rollouts & Success",
            [" & ".join(cells) + r" \\"])

    if "droid_cpu_alignment" in engineering_payloads:
        name = "droid_cpu_alignment"
        stages = engineering_payloads.get("droid_public_construction")
        cpu_rows = []
        labs = [r["workspace_id"].split("_")[1].upper() for r in payloads[name]["rows"]]
        for i, row in enumerate(payloads[name]["rows"]):
            label = labs[i]
            if labs.count(label) > 1:
                label += " (" + str(labs[:i + 1].count(label)) + ")"
            cells = [label, row["execution_status"].lower(),
                     fmt(name, i, "evaluated_reference_frames", digits=0) + "/" + fmt(name, i, "planned_reference_frames", digits=0),
                     fmt(name, i, "translation_cm", digits=2), fmt(name, i, "rotation_deg", digits=2)]
            if stages:
                cells.append({"PASS":"C", "FAIL":"F", "NOT_RUN":"--"}[stages["rows"][i]["gaussian_status"]]
                    + " / " + fmt("droid_public_construction", i, "prepared_instances", digits=0))
            cpu_rows.append(" & ".join(cells) + r" \\")
        generated["droid_cpu_alignment_engineering.tex"] = table(
            "prospective DROID CPU reconstruction/alignment stage. Every declared workspace remains visible. "
            "Frame coverage precedes center RMS (cm) and median rotation residual (degrees) on evaluated robot-reference frames. "
            "Temporal correspondence and metric alignment use TRAIN evidence only. Unavailable values remain --. "
            + ("GS / prep. gives Gaussian training completion (C) and prepared automatic instances; -- is unavailable. "
             "The RPL public-mesh stage fails; completed processing with no discoveries remains zero. "
             "Prepared instances are proposals, not verified distinct objects or accepted simulator assets. "
             "These stages do not establish full simulator builds or autonomous physical success."
             if stages else "A completed CPU stage is not Gaussian reconstruction, a full simulator build, or autonomous physical success."),
            "tab:droid-cpu-alignment", "llcccc" if stages else "llccc",
            "Workspace & CPU state & Eval./planned & cm & deg" + (" & GS / prep." if stages else ""), cpu_rows)

    if "room_common_view" in engineering_payloads:
        from robo.eval.paper_room_diagnostic import METHODS
        name = "room_common_view"
        diagnostic_rows = []
        for method, label in zip(METHODS, ["Raw Gaussian", "Factorized composite"]):
            i = next(i for i, row in enumerate(payloads[name]["rows"]) if row["method"] == method)
            diagnostic_rows.append(" & ".join([label,
                fmt(name, i, "n_scenes", digits=0),
                fmt(name, i, "n_images", digits=0) + "/" +
                fmt("room_common_view_coverage", None, "planned_views_per_method", digits=0),
                fmt(name, i, "psnr", digits=3), fmt(name, i, "ssim", digits=5), fmt(name, i, "lpips", digits=5)]) + r" \\")
        generated["room_common_view_engineering.tex"] = table(
            "fixed common-view appearance diagnostic. Paired/planned retains the original two-scene view denominator. "
            "Raw renders exist for both planned scenes; the factorized room is available for only one. "
            "Both quality rows use all of that same scene's predeclared common TEST cameras. "
            "These are descriptive means, not a scene-population improvement estimate; no confidence interval is interpreted.",
            "tab:room-common-view", "lccccc", "Representation & scenes & paired/planned & PSNR & SSIM & LPIPS",
            diagnostic_rows, wide=True)

    if "full_qualification" in engineering_payloads:
        name = "full_qualification"
        fields = ("planned_scenes", "planned_objects", "planned_semantic_queries",
                  "planned_qualification_cells", "prerequisite_checked_cells",
                  "actual_900_step_cells", "qualified_cells", "policy_executed")
        generated["full_qualification_engineering.tex"] = table(
            "Full-cohort manipulation prerequisites. Selected queries follow the frozen task budget; "
            "queries are shown as selected/input, and logical cells retain unavailable construction/planning outcomes. Checked denotes prerequisite "
            "records; simulated denotes completed 900-step qualification checks. Neither is a policy attempt. "
            "Policy success remains unmeasured; zero executed episodes is not a zero success rate.",
            "tab:full-qualification", "cccccccc",
            "Scenes & Objects & Queries sel./input & Planned cells & Checked & Simulated & Qualified & Rollouts",
            [" & ".join(fmt(name, None, key, digits=0) +
                         ("/" + fmt(name, None, "input_semantic_queries", digits=0)
                          if key == "planned_semantic_queries" else "") for key in fields) + r" \\"], wide=True)

    if "full_appearance" in engineering_payloads:
        name = "full_appearance"
        labels = ("Raw input Gaussian", "Composite, GT discovery", "Composite, automatic discovery", "Harmonizer")
        appearance_rows = []
        for method, label in zip(FIDELITY_ROSTER[:4], labels):
            i = next(i for i, row in enumerate(payloads[name]["rows"]) if row["method"] == method)
            appearance_rows.append(" & ".join([label,
                fmt(name, i, "n_scenes", digits=0) + "/" +
                fmt(name + "_coverage", None, "planned_scenes", digits=0),
                fmt(name, i, "n_images", digits=0) + "/" +
                fmt(name + "_coverage", None, "planned_views_per_method", digits=0),
                fmt(name, i, "psnr", digits=3), fmt(name, i, "ssim", digits=4),
                fmt(name, i, "lpips", digits=4)]) + r" \\")
        generated["object_factorization_main.tex"] = table(
            "Full planned held-out room cohort. Quality uses identical common available TEST views for raw and automatic "
            "composite rows; common/planned preserves unavailable composites. Raw rendering is available for every planned view. "
            "Zero-object scenes with unchanged backgrounds remain included and are not successful interactive simulator builds. "
            "GT-discovery and Harmonizer rows remain unmeasured. Means are descriptive; no paired-difference significance is claimed.",
            "tab:full-appearance", "lccccc",
            "Representation & scenes common/planned & views common/planned & PSNR & SSIM & LPIPS",
            appearance_rows, wide=True)

    if "task_support_closure" in engineering_payloads:
        name = "task_support_closure"
        cells = [fmt(name, None, key, digits=0) for key in
                 ("planned_conditions", "planned_queries", "failed_constructor_query_rows", "invalid_queries")]
        generated["task_support_closure_engineering.tex"] = table(
            "task-support applicability after feature sealing. Missing robot frames and policy cameras "
            "make every query invalid under the fixed requirements. Constructor failures remain included. "
            "Every training and held-out fold has one class; LOSO predictions and all predictive metrics "
            "remain unmeasured, not zero.", "tab:task-support-closure", "ccccc",
            "Conditions & Queries & Build-failed queries & Invalid & LOSO",
            [" & ".join(cells + ["not run"]) + r" \\"])

    for claim_id, decision in config.get("claim_decisions", {}).items():
        if decision.get("status") not in {"PASS", "FAIL", "NOT_RUN"}:
            raise ValueError("claim decision requires PASS, FAIL or NOT_RUN")
        claims.append({"claim_id":claim_id,"paper_file":decision.get("paper_file", ""),
            "source_artifact":"","source_sha256":"","source_field":"",
            "value_in_source":None,"value_in_text":"","status":decision.get("decision", ""),
            "claim_gate":decision["status"],"owner":"E9", **{key:decision.get(key, "") for key in
                ("required_experiment","required_gate","observed_result","enabled_sentence","removed_sentence")}})
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".paper-audit-", dir=out_dir.parent))
    try:
        (stage / "generated_tables").mkdir()
        (stage / "sources").mkdir()
        for name, payload in payloads.items():
            write_json(stage / "sources" / f"{name}.json", payload)
            data = payload.get("rows", [payload])
            fields = sorted({k for row in data for k in row})
            with (stage / "sources" / f"{name}.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        for filename, tex in generated.items():
            (stage / "generated_tables" / filename).write_text(tex)
        with (stage / "claim_ledger.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in claims for k in row)))
            writer.writeheader()
            writer.writerows(claims)
        decisions = config.get("claim_decisions", {})
        (stage / "claim_decisions.yaml").write_text(yaml.safe_dump(decisions))
        report = {"freeze_id": config["freeze_id"], "mode": "preliminary_audit",
                  "paper_ready": False, "config": str(config_path.resolve()),
                  "formatter_commit": subprocess.check_output(
                      ["git", "rev-parse", "HEAD"],
                      cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                  "config_sha256": _sha256(config_path), "sources": sources,
                  "source_freezes_are_distinct": True,
                  "tables": {name: {"sha256": _sha256(stage / "generated_tables" / name)}
                             for name in generated},
                  "limitations": "Preliminary source-bound display; not a common full experiment freeze.",
                  "runtime_display": ("attributed_algorithm_phase" if agentic_context and agentic_context["full_cohort"] else
                                      "omitted_incomplete_generation_accounting" if agentic_context else "legacy_scope"),
                  "runtime_scope_note": agentic_context.get("runtime_scope_note") if agentic_context else None,
                  "omitted_unmeasured_object_table": omit_objects,
                  "unmeasured_object_skeleton": "object_fidelity_unmeasured_skeleton.tex" if omit_objects else None,
                  "object_method_source_rows_preserved": len([r for r in fidelity["rows"] if r["unit"] == "object"]),
                  "prose_decisions_applied": False}
        if make_paired_figure:
            from robo.eval.paper_figures import render_agentic_paired
            report["figures"] = render_agentic_paired(payloads["agentic_paired_uncertainty"],
                sources["agentic_paired_uncertainty"], stage / "generated_figures")
        write_json(stage / "paper_table_provenance.json", report)
        write_json(stage / "submission_audit.json", {
            "submission_freeze": "FAIL", "reason": "Preliminary sources and incomplete required experiments",
            "paper_ready": False, "claim_decisions": decisions})
        shutil.copy2(config_path, stage / "resolved_config.yaml")
        stage.rename(out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for filename in generated:
        _copy(out_dir / "generated_tables" / filename, paper_root, f"tables/{filename}")
    for filename in report.get("figures", {}):
        _copy(out_dir / "generated_figures" / filename, paper_root, f"figures/{filename}")
    if paper_root:
        audit_dir = paper_root / "audit"
        audit_dir.mkdir(exist_ok=True)
        for filename in ("paper_table_provenance.json", "claim_ledger.csv", "claim_decisions.yaml"):
            shutil.copy2(out_dir / filename, audit_dir / filename)
    return report


def generate(config_path: str | Path, out_dir: str | Path,
             paper_root: str | Path | None = None) -> dict:
    config_path, out_dir = Path(config_path), Path(out_dir)
    paper_root = Path(paper_root) if paper_root else None
    config = yaml.safe_load(config_path.read_text())
    if "native_scale_up" in config:
        if set(config) != {"native_scale_up"}:
            raise ValueError("native_scale_up must be a separate pipeline configuration")
        from robo.eval.native_scale_tables import generate as generate_native
        return generate_native(config_path, out_dir, paper_root=paper_root)
    if config.get("audited_artifacts"):
        return _audited_tables(config, config_path, out_dir, paper_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "tables": {}, "paper_copies": {}}

    job = config.get("construction")
    if job:
        directory = out_dir / "construction"
        report["tables"]["construction"] = construction_metrics.generate(
            job["input"], directory)
        report["paper_copies"]["construction"] = _copy(
            directory / "construction_table.tex", paper_root,
            "tables/automatic_construction_main.tex")

    job = config.get("fidelity")
    if job:
        directory = out_dir / "fidelity"
        report["tables"]["fidelity"] = fidelity_metrics.evaluate_manifest(
            job["manifest"], directory,
            lpips_device=job.get("lpips_device", "cpu"))
        # Fidelity JSON/CSV are authoritative. The mixed-unit table remains a
        # fixed paper skeleton until all methods exist, to avoid dropping rows.

    job = config.get("manipulation")
    if job:
        directory = out_dir / "manipulation"
        report["tables"]["manipulation"] = main_table.generate(
            job["harness_config"], job["ledger"], directory)
        report["paper_copies"]["manipulation"] = _copy(
            directory / "main_table.tex", paper_root,
            "tables/frozen_policy_main.tex")

    job = config.get("audit")
    if job:
        directory = out_dir / "audit"
        prediction_dir = directory / "predictions"
        audit_loso.generate(job["features"], prediction_dir,
                            l2=float(job.get("l2", 1.0)))
        report["tables"]["audit"] = audit_metrics.generate(
            prediction_dir / "heldout_predictions.csv", directory)
        report["paper_copies"]["audit"] = _copy(
            directory / "audit_table.tex", paper_root,
            "tables/task_local_audit_main.tex")

    job = config.get("harmony_visual")
    if job:
        directory = out_dir / "harmony_visual"
        report["tables"]["harmony_visual"] = harmony_visual_metrics.generate(
            job["manifest"], directory,
            lpips_device=job.get("lpips_device", "cpu"))
        report["paper_copies"]["harmony_visual"] = _copy(
            directory / "harmony_visual_table.tex", paper_root,
            "tables/harmony_visual_main.tex")

    job = config.get("real_world")
    if job:
        directory = out_dir / "real_world"
        report["tables"]["real_world"] = real_world_table.generate(
            job["construction"], job.get("trials"), directory)
        report["paper_copies"]["real_world"] = _copy(
            directory / "real_world_table.tex", paper_root,
            "tables/real_world_main.tex")

    write_json(out_dir / "paper_table_provenance.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--bundle-existing", help="Package already generated tables without rerunning metrics")
    parser.add_argument("--publication-receipt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-root")
    args = parser.parse_args(argv)
    if args.bundle_existing:
        if args.config or not args.paper_root or not args.publication_receipt:
            parser.error("existing bundle requires paper-root and publication-receipt, and no config")
        from robo.eval.paper_bundle import package
        result = package(args.bundle_existing, args.paper_root, args.publication_receipt, args.out)
    else:
        if not args.config or args.publication_receipt:
            parser.error("generation requires config; publication-receipt is only for bundle-existing")
        result = generate(args.config, args.out, args.paper_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
