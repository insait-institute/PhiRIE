"""Final tables as checked projections of existing canonical native ledgers.

No outcome is inferred from component completion. Missing engine work is not
zero success; construction failures have zero service success and unknown
policy success. Paired inference reuses the established hierarchical bootstrap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

from robo.campaign.core import checked_path, digest, load, receipt, rows, save

SHA = re.compile(r"^[0-9a-f]{64}$")
UNMEASURED = {"NOT_SCHEDULED", "ENVIRONMENT_FAILED", "CODE_FAILED", "RESOURCE_FAILED"}
BUILD = {"BUILD_FAILED", "ABSTAINED"}


def unique(records: list[dict], key: str) -> dict:
    result = {}
    for r in records:
        if r[key] in result:
            raise ValueError(f"duplicate {key}")
        result[r[key]] = r
    return result


def missing(unit: dict, reason: str) -> dict:
    return {**unit, "terminal_status": reason, "measured": False, "executed": False,
            "success": None, "service_success": None}


def validate_block(contract: dict, index: dict) -> list[dict]:
    """Read-only join of final obligations, canonical plan, ledger and artifacts."""
    if index.get("block_id") != contract["block_id"]:
        raise ValueError("result index names a different block")
    if load(checked_path(index["contract"])) != contract:
        raise ValueError("index contract differs")
    plan = unique(rows(checked_path(index["native_plan"])), "unit_id")
    ledger = unique(rows(checked_path(index["native_ledger"])), "unit_id")
    bindings = unique(rows(checked_path(index["unit_bindings"])), "final_unit_id")
    expected = unique(contract["units"], "final_unit_id")
    if set(bindings) != set(expected):
        raise ValueError("crosswalk must preserve the ENTIRE planned block")
    native_ids = [r["native_unit_id"] for r in bindings.values()]
    if len(set(native_ids)) != len(native_ids) or set(native_ids) != set(plan):
        raise ValueError("native plan/crosswalk must be a bijection, not a successful subset")
    if set(ledger) - set(plan):
        raise ValueError("unplanned native output")
    outputs, engines, policies = [], set(), set()
    execution_bindings = {}
    scalar_keys = ("canonical_instance_id", "task_id", "policy_rng_seed", "native_horizon",
                   "controller_method", "scope", "sensor_regime", "renderer", "split")
    for fid, u in expected.items():
        b = bindings[fid]
        n = b["native_unit_id"]
        p = plan[n]
        if (any(p.get(k) != u[k] for k in scalar_keys) or
                str(p.get("reset_id")) != u["reset_id"] or str(p.get("layout_id")) != u["layout_id"]):
            raise ValueError("native plan does not implement the declared final intervention")
        r = ledger.get(n)
        if r is None:
            outputs.append(missing(u, "MISSING_LEDGER_ROW")); continue
        if any(p.get(k) != r.get(k) for k in scalar_keys + ("reset_id", "layout_id", "cohort_id", "policy_id")):
            raise ValueError("canonical ledger identity differs from canonical plan")
        status = r.get("terminal_status")
        if status in UNMEASURED:
            if r.get("success") is not None or r.get("executed") not in (None, False):
                raise ValueError("unmeasured engineering work carries an outcome")
            outputs.append(missing(u, status)); continue
        if status in BUILD:
            if u["arm_id"] == "REF_NATIVE":
                raise ValueError("reference unavailability is an environment failure, not a method build failure")
            if r.get("executed") is not False or r.get("success") is not None:
                raise ValueError("unexecuted method failure has no native policy success")
            failure = load(checked_path(b["failure_evidence"]))
            if failure.get("native_unit_id") != n or failure.get("classification") != status:
                raise ValueError("failure evidence does not bind the native unit/status")
            outputs.append({**u, "native_unit_id": n, "terminal_status": status,
                            "measured": True, "executed": False, "success": None,
                            "service_success": False})
            continue
        if status != "RECORDED" or r.get("executed") is not True or type(r.get("success")) is not bool:
            raise ValueError("invalid canonical recorded episode")
        result = load(checked_path({"path": r["result_path"], "sha256": r["result_sha256"]}))
        if (result.get("executed") is not True or result.get("success") != r["success"] or
                result.get("error") is not None or result.get("execution_kind") != "closed_loop_visual_policy"):
            raise ValueError("record is not a valid learned-policy outcome")
        policy = result.get("policy_identity", {})
        engine = policy.get("policy_engine", result.get("policy_engine", {}))
        if not engine.get("engine_id") or not engine.get("process_uuid"):
            raise ValueError("missing real policy process identity")
        engines.add((engine["engine_id"], engine["process_uuid"]))
        policies.add(digest(policy))
        execution = load(checked_path(b["execution_contract"]))
        if execution.get("native_unit_id") != n or execution.get("result_sha256") != r["result_sha256"]:
            raise ValueError("execution contract not tied to native result bytes")
        for field in ("policy_contract", "camera_contract", "task_contract", "reset_contract", "physics"):
            checked_path(execution[field])
        if execution.get("recipe_sha256") != contract["recipe_sha256"]:
            raise ValueError("executed recipe differs from frozen recipe")
        if u["arm_id"] != "REF_NATIVE" and u["renderer"] != "native":
            sync = load(checked_path(execution["state_sync"]))
            if (sync.get("passed") is not True or sync.get("measurement_kind") != "real_runtime" or
                    type(sync.get("frames_checked")) is not int or sync["frames_checked"] < 1 or
                    sync.get("max_state_lag_ticks") != 0 or sync.get("silent_fallback_frames") != 0):
                raise ValueError("Gaussian/enhanced policy arm lacks current-state observation evidence")
        if u["arm_id"] in ("SIMFOUNDRY_ADAPTED", "POLARIS_ADAPTED"):
            method = load(checked_path(execution["official_method"]))
            required = "SimFoundry" if u["arm_id"].startswith("SIMFOUNDRY") else "PolaRiS"
            if (method.get("method") != required or method.get("adaptation") != "common_importer" or
                    not re.fullmatch(r"[0-9a-f]{40}", str(method.get("upstream_commit", ""))) or
                    type(method.get("manual_minutes")) not in (int, float) or
                    not math.isfinite(method["manual_minutes"]) or method["manual_minutes"] < 0 or
                    not method.get("acquisition")):
                raise ValueError("official method identity/effort/adaptation is missing")
        execution_bindings[(u["reset_id"], u["arm_id"])] = execution
        outputs.append({**u, "native_unit_id": n, "terminal_status": status,
                        "measured": True, "executed": True, "success": r["success"],
                        "service_success": r["success"], "result": receipt(r["result_path"])})
    if len(engines) > 1 or len(policies) > 1:
        raise ValueError("paired block mixed policy processes/checkpoints")
    for reset in {u["reset_id"] for u in contract["units"]}:
        available = {a: x for (rid, a), x in execution_bindings.items() if rid == reset}
        ref = available.get("REF_NATIVE")
        if any(a != "REF_NATIVE" for a in available) and ref is None:
            raise ValueError("recorded method missing its same-process reference")
        if ref is None:
            continue
        for x in available.values():
            for k in ("policy_contract", "camera_contract", "task_contract", "reset_contract"):
                if x[k]["sha256"] != ref[k]["sha256"]:
                    raise ValueError(f"paired {k} changed")
        if contract["suite"]["vary"] == "observation":
            ours = [x for a, x in available.items() if a != "REF_NATIVE"]
            if len({x["physics"]["sha256"] for x in ours}) > 1:
                raise ValueError("observation ablation changed physical assets or state")
    return outputs


def aggregate(records: list[dict]) -> dict:
    measured = [r for r in records if r["measured"]]
    executed = [r for r in measured if r["executed"]]
    s = sum(r["service_success"] for r in measured)
    complete = len(measured) == len(records) and bool(records)
    return {"planned": len(records), "measured": len(measured), "executed": len(executed),
            "successes_observed": s, "build_failed": sum(r["terminal_status"] == "BUILD_FAILED" for r in records),
            "abstained": sum(r["terminal_status"] == "ABSTAINED" for r in records),
            "unmeasured": len(records)-len(measured), "complete": complete,
            "instances": len({r["case_id"] for r in records}),
            "layouts": len({r["layout_id"] for r in records}),
            "success_per_planned": s/len(records) if complete else None,
            "success_per_executed": s/len(executed) if executed else None,
            "execution_coverage": len(executed)/len(records) if records else None}


def contrasts(records: list[dict], suites: list[dict]) -> list[dict]:
    from robo.eval.metric_utils import paired_hierarchical_bootstrap
    result = []
    for suite in suites:
        group = [r for r in records if r["suite"] == suite["id"]]
        for a, b in suite.get("contrasts", []):
            aa = {(r["case_id"], r["reset_id"]): r for r in group if r["arm_id"] == a}
            bb = {(r["case_id"], r["reset_id"]): r for r in group if r["arm_id"] == b}
            complete = bool(aa) and set(aa) == set(bb) and all(
                aa[k]["measured"] and bb[k]["measured"] for k in aa)
            pairs = [{"layout_id": aa[k]["layout_id"], "canonical_instance_id": aa[k]["canonical_instance_id"],
                      "reset_id": aa[k]["reset_id"], "a": int(aa[k]["service_success"]),
                      "b": int(bb[k]["service_success"])} for k in sorted(set(aa) & set(bb))
                     if aa[k]["measured"] and bb[k]["measured"]]
            stats = paired_hierarchical_bootstrap(pairs)
            supported = complete and stats["layouts"] >= 2
            result.append({"suite": suite["id"], "reference": a, "method": b,
                           "paired_complete": complete, "matched_resets": len(pairs),
                           "delta_pp": 100*stats["delta"] if supported else None,
                           "ci95_pp": [100*v for v in stats["ci95"]] if supported else [None, None],
                           "positive_improvement_supported": bool(supported and stats["ci95"][0] > 0),
                           "reference_has_success": any(p["a"] for p in pairs),
                           "preservation_or_equivalence": "NOT_TESTED",
                           "partial_pairs_descriptive_only": stats,
                           "interpretation": "complete registered contrast; no cross-engine ranking"})
    return result


def csv_write(path: Path, rs: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rs for k in r))
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rs)


def release(root: str, index_paths: list[str], out: str) -> dict:
    from robo.campaign.finalize import verify
    plan = verify(root)
    protocol = load(checked_path(plan["protocol"]))
    by_block = {b["block_id"]: b for b in plan["blocks"]}
    indexes = {}
    for p in index_paths:
        index = load(p)
        bid = index["block_id"]
        if bid not in by_block or bid in indexes:
            raise ValueError("unexpected/repeated block result index")
        if index["contract"] != by_block[bid]["contract"]:
            raise ValueError("result refers to another frozen block")
        indexes[bid] = (index, receipt(p))
    projected = []
    for bid, entry in by_block.items():
        block = load(checked_path(entry["contract"]))
        if bid in indexes:
            projected.extend(validate_block(block, indexes[bid][0]))
        else:
            projected.extend(missing(u, "UNBOUND_OR_MISSING_RESULT") for u in block["units"])
    expected = unique(rows(checked_path(plan["expected_units"])), "final_unit_id")
    actual = unique(projected, "final_unit_id")
    if set(actual) != set(expected):
        raise ValueError("release dropped or added planned units")
    groups = defaultdict(list)
    per_task_groups = defaultdict(list)
    for r in projected:
        groups[(r["suite"], r["arm_id"])].append(r)
        per_task_groups[(r["suite"], r["arm_id"], r["task_id"])].append(r)
    tables = [{"suite": s, "arm": a, **aggregate(rs)} for (s, a), rs in sorted(groups.items())]
    per_task = [{"suite": s, "arm": a, "task": t, **aggregate(rs)}
                for (s, a, t), rs in sorted(per_task_groups.items())]
    comparisons = contrasts(projected, protocol["suites"])
    ready = all(r["complete"] for r in tables)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=False)
    csv_write(outdir/"native_tables.csv", tables)
    csv_write(outdir/"per_task.csv", per_task)
    save(outdir/"comparisons.json", comparisons)
    # This is an analysis projection; original ledgers remain authoritative.
    save(outdir/"analysis_projection.jsonl", projected, jsonl=True)
    summary = {"status": "MEASUREMENTS_COMPLETE" if ready else "PARTIAL",
               "planned_units": len(projected), "measured_units": sum(r["measured"] for r in projected),
               "executed_episodes": sum(r["executed"] for r in projected),
               "missing_blocks": [b for b in by_block if b not in indexes],
               "planned_blocks": len(by_block), "linked_blocks": len(indexes),
               "native_data_complete": ready, "paper_ready": "REQUIRES_SCIENTIFIC_AND_LAYOUT_REVIEW",
               "historical_data_not_pooled": True, "result_indexes": [r for _, r in indexes.values()],
               "source_plan": receipt(Path(root)/"plan.json"),
               "previously_observed_instances": sum(r["prior_outcomes_seen"] for r in rows(checked_path(plan["cases"]))) }
    save(outdir/"summary.json", summary)
    lines = ["# Final experiment release", "", f"Status: **{summary['status']}**", "",
             f"Planned units: {len(projected)}. Measured: {summary['measured_units']}. "
             f"Actual learned-policy episodes: {summary['executed_episodes']}.", "",
             "| Suite | Arm | Executed/planned | Success/planned | Unmeasured |",
             "|---|---|---:|---:|---:|"]
    for r in tables:
        score = f"{r['success_per_planned']:.1%}" if r["complete"] else "NOT_COMPLETE"
        lines.append(f"| {r['suite']} | {r['arm']} | {r['executed']}/{r['planned']} | {score} | {r['unmeasured']} |")
    lines += ["", "Completion is not superiority. Consult the paired contrasts, retained-context",
              "inventory, visual diagnostics and original canonical ledgers before enabling claims.",
              "No automatic real-world, universal-room, or equivalence claim is enabled."]
    (outdir/"AGENT_REPORT.md").write_text("\n".join(lines)+"\n")
    files = [p for p in sorted(outdir.iterdir()) if p.is_file()]
    (outdir/"SHA256SUMS.txt").write_text("".join(f"{receipt(p)['sha256']}  {p.name}\n" for p in files))
    return summary
