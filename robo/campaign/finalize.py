"""Bound the final submission campaign; reuse campaign.runner for execution.

This module does not create a simulator, a policy, or a second rollout ledger.
It seals an intended roster independently of readiness, binds actual existing
workers, and refuses to declare the successfully bound subset a complete study.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .core import checked_path, digest, load, receipt, rows, safe_id, save

TASKS = ("PickPlaceCounterToSink", "PickPlaceSinkToCounter")
HORIZONS = {"PickPlaceCounterToSink": 600, "PickPlaceSinkToCounter": 900}
SCHEMA = "simanyroom-final-experiments-v1"


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.startswith(("RESOLVE", "/REPLACE")):
        raise ValueError(f"missing {name}")
    return value


def validate_protocol(p: dict) -> None:
    if p.get("schema") != SCHEMA:
        raise ValueError("unknown final protocol")
    safe_id(p["study_id"])
    for key in ("layouts", "instances_per_task", "resets"):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError(f"positive integer {key} required")
    if p.get("tasks") != list(TASKS):
        raise ValueError("the final scope uses the two declared rigid task families")
    if p.get("sensor") != "rgb_video_public_marker_v1":
        raise ValueError("declare RGB video and its public calibration separately")
    if not p.get("suites") or len({s['id'] for s in p['suites']}) != len(p['suites']):
        raise ValueError("nonempty unique suites required")
    for s in p["suites"]:
        safe_id(s["id"])
        if s["membership"] not in ("all", "one_per_layout_task"):
            raise ValueError("unsupported outcome-independent subset")
        arms = s["arms"]
        ids = [safe_id(a["id"]) for a in arms]
        if len(set(ids)) != len(ids) or len(ids) < 2 or ids[0] != "REF_NATIVE":
            raise ValueError("each paired suite starts with a unique fresh reference")
        if s.get("vary") not in ("construction", "observation", "scope"):
            raise ValueError("declare the treatment axis")
        for a in arms:
            for k in ("controller_method", "renderer", "scope"):
                nonempty(a.get(k), k)
        for a, b in s.get("contrasts", []):
            if a == b or a not in ids or b not in ids:
                raise ValueError("contrast must name two different declared arms")
    if type(p.get("max_active_jobs")) is not int or p["max_active_jobs"] < 1:
        raise ValueError("positive global scheduling cap required")


def select(pool_path: str, protocol_path: str, out: str) -> dict:
    """Choose source instances before construction; never filter by readiness."""
    p = load(protocol_path)
    validate_protocol(p)
    source = rows(pool_path)
    seen = set()
    groups: dict[str, dict[str, list[dict]]] = {}
    for r in source:
        cid = safe_id(r["case_id"])
        if cid in seen:
            raise ValueError("duplicate source case")
        seen.add(cid)
        nonempty(r.get("canonical_instance_id"), "canonical_instance_id")
        if r.get("split") != "test" or r.get("dataset") != "robocasa":
            raise ValueError("source pool must be the declared RoboCasa test roster")
        if type(r.get("prior_outcomes_seen")) is not bool:
            raise ValueError("declare prior outcome exposure, including exploratory follow-ups")
        if any(k in r for k in ("success", "psnr", "method_score", "build_pass", "quality")):
            raise ValueError("method outcomes cannot enter source selection")
        if r["task_id"] not in TASKS:
            raise ValueError("task outside final scope")
        checked_path(r["source_index"])
        groups.setdefault(str(r["layout_id"]), {}).setdefault(r["task_id"], []).append(r)
    eligible = [g for g, v in groups.items()
                if all(len(v.get(t, [])) >= p["instances_per_task"] for t in TASKS)]
    eligible.sort(key=lambda g: digest([p["selection_seed"], "layout", g]))
    if len(eligible) < p["layouts"]:
        raise ValueError("insufficient SOURCE families; record a prospective amendment, not fake IDs")
    chosen = []
    for g in eligible[:p["layouts"]]:
        for t in TASKS:
            candidates = sorted(groups[g][t], key=lambda r: digest([p["selection_seed"], r["case_id"]]))
            chosen.extend(candidates[:p["instances_per_task"]])
    if len({r["canonical_instance_id"] for r in chosen}) != len(chosen):
        raise ValueError("canonical instances duplicated across source slots")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    save(target / "cases.jsonl", chosen, jsonl=True)
    save(target / "selection.json", {
        "source": receipt(pool_path), "protocol": receipt(protocol_path),
        "counts": {"pool": len(source), "selected": len(chosen), "layouts": p["layouts"]},
        "selection": "source availability and stable hash only; no construction gates",
        "source_families_below_quota": sorted(set(groups) - set(eligible)),
        "previously_observed_cases": sum(r["prior_outcomes_seen"] for r in chosen),
    })
    return {"cases": len(chosen), "inventory": str(target / "cases.jsonl")}


def freeze(cases_path: str, protocol_path: str, recipe_path: str, out: str) -> dict:
    p, recipe = load(protocol_path), load(recipe_path)
    validate_protocol(p)
    cases = rows(cases_path)
    n = p["layouts"] * len(TASKS) * p["instances_per_task"]
    if len(cases) != n or len({r["case_id"] for r in cases}) != n:
        raise ValueError("case roster differs from fixed quota")
    if len({r["canonical_instance_id"] for r in cases}) != n:
        raise ValueError("canonical instance duplicated")
    for r in cases:
        safe_id(r["case_id"])
        checked_path(r["source_index"])
        if r.get("dataset") != "robocasa" or r.get("split") != "test" or type(r.get("prior_outcomes_seen")) is not bool:
            raise ValueError("dataset/split/exposure missing")
    groups: dict[tuple, list[dict]] = {}
    for r in cases:
        groups.setdefault((str(r["layout_id"]), r["task_id"]), []).append(r)
    if (len({g[0] for g in groups}) != p["layouts"] or
            len(groups) != p["layouts"] * len(TASKS) or
            any(g[1] not in TASKS or len(rs) != p["instances_per_task"] for g, rs in groups.items())):
        raise ValueError("layout/task strata differ from final plan")
    if recipe.get("frozen_before_outcomes") is not True or recipe.get("test_tuning") is not False:
        raise ValueError("default recipe must be selected on DEV, not final outcomes")
    for key in ("generator", "segmentation", "inpainting", "harmonizer", "calibration", "policy"):
        nonempty(recipe.get(key), f"recipe.{key}")
    for key in ("model_lock", "dev_selection", "policy_contract"):
        checked_path(recipe[key])
    root = Path(out).absolute()
    root.mkdir(parents=True, exist_ok=False)
    subset = {min(rs, key=lambda r: digest([p["selection_seed"], r["case_id"]]))["case_id"]
              for rs in groups.values()}
    blocks, units = [], []
    for suite in p["suites"]:
        for r in sorted(cases, key=lambda r: r["case_id"]):
            if suite["membership"] == "one_per_layout_task" and r["case_id"] not in subset:
                continue
            bid = safe_id(suite["id"] + "-" + r["case_id"])
            required = []
            for reset in range(p["resets"]):
                for arm in suite["arms"]:
                    u = {"study_id": p["study_id"], "suite": suite["id"], "block_id": bid,
                         "case_id": r["case_id"], "canonical_instance_id": r["canonical_instance_id"],
                         "layout_id": str(r["layout_id"]), "task_id": r["task_id"],
                         "reset_id": str(reset), "policy_rng_seed": p["policy_seed_base"] + reset,
                         "native_horizon": HORIZONS[r["task_id"]], "arm_id": arm["id"],
                         "controller_method": arm["controller_method"], "renderer": arm["renderer"],
                         "scope": arm["scope"], "sensor_regime": p["sensor"], "split": "test",
                         "prior_outcomes_seen": r["prior_outcomes_seen"]}
                    u["final_unit_id"] = digest(u)[:24]
                    required.append(u)
            block = {"schema": SCHEMA, "block_id": bid, "suite": suite,
                     "case": r, "recipe_sha256": digest(recipe), "units": required,
                     "fresh_reference": True, "one_policy_process": True}
            path = root / "blocks" / (bid + ".json")
            save(path, block)
            blocks.append({"block_id": bid, "contract": receipt(path), "units": len(required)})
            units.extend(required)
    save(root / "protocol.json", p)
    save(root / "recipe.json", recipe)
    save(root / "cases.jsonl", cases, jsonl=True)
    save(root / "expected_units.jsonl", units, jsonl=True)
    plan = {"schema": SCHEMA, "protocol": receipt(root / "protocol.json"),
            "recipe": receipt(root / "recipe.json"), "cases": receipt(root / "cases.jsonl"),
            "expected_units": receipt(root / "expected_units.jsonl"), "blocks": blocks,
            "recipe_dependencies": {k: recipe[k] for k in ("model_lock", "dev_selection", "policy_contract")},
            "planned_units": len(units), "planned_blocks": len(blocks),
            "scientific_execution": False}
    save(root / "plan.json", plan)
    return {"plan": str(root / "plan.json"), "units": len(units), "blocks": len(blocks)}


def verify(root: str | Path) -> dict:
    plan = load(Path(root) / "plan.json")
    if plan.get("schema") != SCHEMA:
        raise ValueError("invalid final plan")
    for key in ("protocol", "recipe", "cases", "expected_units"):
        checked_path(plan[key])
    for r in plan["recipe_dependencies"].values():
        checked_path(r)
    for b in plan["blocks"]:
        checked_path(b["contract"])
    return plan


def admit(identity_path: str, rubric_path: str, reference_path: str, comparison_path: str, out: str) -> dict:
    """Wrap actual existing DEV evidence for command admission, without changing tolerances."""
    identity, rubric = load(identity_path), load(rubric_path)
    if identity.get("passed") is not True or rubric.get("passed") is not True:
        raise ValueError("existing identity/rubric checks must genuinely pass")
    ref, cmp = load(reference_path), load(comparison_path)
    for r in (ref, cmp):
        if (r.get("executed") is not True or type(r.get("success")) is not bool or
                r.get("error") is not None or r.get("execution_kind") != "closed_loop_visual_policy"):
            raise ValueError("admission requires two actual visual-policy DEV episodes")
    if ref.get("split") not in ("development", "dev") or cmp.get("split") not in ("development", "dev"):
        raise ValueError("admission data must be explicitly DEV, never TEST")
    pi = ref.get("policy_identity", {})
    engine = pi.get("policy_engine", ref.get("policy_engine", {}))
    if not engine.get("process_uuid") or digest(pi) != digest(cmp.get("policy_identity", {})):
        raise ValueError("DEV pair must share the actual policy process")
    target = Path(out); target.mkdir(parents=True, exist_ok=False)
    evidence = {"identity": receipt(identity_path), "rubric": receipt(rubric_path),
                "reference": receipt(reference_path), "comparison": receipt(comparison_path)}
    gate = {"passed": True, "measurement_kind": "real_native_dev", "evidence": evidence,
            "requires_positive_task_success": False, "unchanged_native_thresholds": True}
    save(target/"same_engine_gate.json", gate)
    save(target/"rubric_gate.json", gate)
    return {"same_engine_gate": receipt(target/"same_engine_gate.json"),
            "rubric_gate": receipt(target/"rubric_gate.json")}


def prepare(root: str, bindings_path: str, runtime_path: str, out: str) -> dict:
    """Bind admitted command tasks while explicitly retaining every unbound block."""
    from .runner import prepare as prepare_runner
    plan = verify(root)
    by_id = {b["block_id"]: b for b in plan["blocks"]}
    bindings = rows(bindings_path)
    seen, tasks = set(), []
    for b in bindings:
        bid = b["block_id"]
        if bid not in by_id or bid in seen:
            raise ValueError("unexpected or duplicate final block binding")
        seen.add(bid)
        task = load(checked_path(b["task"]))
        expected = by_id[bid]["contract"]
        if task["id"] != bid or task["kind"] != "command" or task.get("split") != "test":
            raise ValueError("bind the actual paired command, not a component/model call")
        params = task["params"]
        if params.get("purpose") != "native_policy" or params.get("fresh_reference") is not True:
            raise ValueError("fresh native policy block required")
        if task.get("sensor") != "rgb_video" or task.get("camera_source") != "estimated":
            raise ValueError("final video protocol cannot be relabeled posed RGB-D")
        if task.get("inputs", {}).get("final_contract") != expected:
            raise ValueError("worker must bind the exact final contract")
        if not any("{final_contract}" in s for s in params["argv"]):
            raise ValueError("worker argv must consume final_contract")
        for key in ("same_engine_gate", "rubric_gate"):
            gate = load(checked_path(params[key]))
            if gate.get("passed") is not True or gate.get("measurement_kind") != "real_native_dev":
                raise ValueError("real native DEV gates required; synthetic test is not admission")
        tasks.append(task)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=False)
    save(out_path / "unbound.json", [{"block_id": b, "planned_units": by_id[b]["units"],
         "status": "UNBOUND_NOT_MEASURED"} for b in by_id if b not in seen])
    save(out_path / "tasks.jsonl", tasks, jsonl=True)
    result = {"bound": len(tasks), "required": len(by_id), "unbound": len(by_id) - len(tasks),
              "full_ready": len(tasks) == len(by_id)}
    if tasks:
        prepare_runner(out_path / "tasks.jsonl", runtime_path, out_path / "jobs")
    save(out_path / "binding.json", {**result, "final_plan": receipt(Path(root) / "plan.json"),
         "bindings": receipt(bindings_path), "tasks": receipt(out_path / "tasks.jsonl")})
    return result


def dispatch(registry_path: str, *, submit: bool = False, max_jobs: int = 1) -> list:
    """Global cap across registered bundles, with a single cross-agent lock.

    Register old/new project bundles before use. Uncertain submissions block
    automatic resubmission and must be reconciled against sacct by the operator.
    """
    from .runner import launch
    registry = load(registry_path)
    cap = registry["max_active_jobs"]
    if type(cap) is not int or cap < 1 or type(max_jobs) is not int or max_jobs < 1:
        raise ValueError("positive caps required")
    bundles = [Path(p).absolute() for p in registry["bundles"]]
    if not bundles or len(set(bundles)) != len(bundles):
        raise ValueError("register unique existing bundles")
    for p in bundles:
        if not (p / "plan.json").is_file():
            raise ValueError(f"bundle has no plan: {p}")
    lockfile = Path(registry_path).absolute().with_suffix(".dispatch.lock")
    with lockfile.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        slots = min(max_jobs, cap)
        if submit:
            active = set(subprocess.check_output(
                ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%A"], text=True).split())
            owned = set(str(i) for i in registry.get("additional_active_job_ids", []))
            for bundle in bundles:
                for intent in (bundle / "jobs").glob("*/intent.json"):
                    sub = intent.parent / "submission.json"
                    if not sub.exists():
                        raise RuntimeError("uncertain submission intent: reconcile scheduler before another launch")
                    r = load(sub)
                    if r.get("returncode") != 0 or not str(r.get("job_id", "")).isdigit():
                        raise RuntimeError("failed/ambiguous submission needs operator reconciliation")
                    owned.add(str(r["job_id"]))
            slots = min(slots, max(0, cap - len(active & owned)))
            account_cap = registry.get("account_active_job_ceiling")
            if account_cap is not None:
                if type(account_cap) is not int or account_cap < 1:
                    raise ValueError("invalid account active-job ceiling")
                slots = min(slots, max(0, account_cap - len(active)))
        report = []
        for b in bundles:
            if len(report) >= slots:
                break
            report.extend(launch(b, submit=submit, max_jobs=slots-len(report)))
        return report


def link_block(contract_path: str, native_plan_path: str, ledger_path: str,
               bindings_path: str, out: str) -> dict:
    """Index real canonical ledger rows; never rewrite outcomes or method labels.

    bindings maps final_unit_id to native_unit_id plus execution receipts. This
    explicit crosswalk supports existing producers without inventing a new ledger.
    """
    from robo.eval.final_release import validate_block
    c = load(contract_path)
    result = {"schema": SCHEMA, "block_id": c["block_id"], "contract": receipt(contract_path),
              "native_plan": receipt(native_plan_path), "native_ledger": receipt(ledger_path),
              "unit_bindings": receipt(bindings_path)}
    validate_block(c, result)
    save(Path(out) / "final_result_index.json", result)
    artifacts = {"final_result_index": receipt(Path(out) / "final_result_index.json")}
    save(Path(out) / "native_outputs.json", {"artifacts": artifacts})
    return result


def collect(root: str, result_index_paths: list[str], out: str) -> dict:
    from robo.eval.final_release import release
    return release(root, result_index_paths, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    p = sub.add_parser("select")
    p.add_argument("--pool", required=True); p.add_argument("--protocol", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("freeze")
    for k in ("cases", "protocol", "recipe", "out"): p.add_argument("--"+k, required=True)
    p = sub.add_parser("verify"); p.add_argument("--root", required=True)
    p = sub.add_parser("admit")
    for k in ("identity", "rubric", "reference", "comparison", "out"): p.add_argument("--"+k, required=True)
    p = sub.add_parser("prepare")
    for k in ("root", "bindings", "runtime", "out"): p.add_argument("--"+k, required=True)
    p = sub.add_parser("dispatch")
    p.add_argument("--registry", required=True); p.add_argument("--submit", action="store_true"); p.add_argument("--max-jobs", type=int, default=1)
    p = sub.add_parser("link")
    for k in ("contract", "native-plan", "ledger", "bindings", "out"): p.add_argument("--"+k, required=True)
    p = sub.add_parser("collect")
    p.add_argument("--root", required=True); p.add_argument("--index", action="append", default=[]); p.add_argument("--out", required=True)
    a = parser.parse_args()
    if a.op == "select": r = select(a.pool, a.protocol, a.out)
    elif a.op == "freeze": r = freeze(a.cases, a.protocol, a.recipe, a.out)
    elif a.op == "verify": r = verify(a.root)
    elif a.op == "admit": r = admit(a.identity, a.rubric, a.reference, a.comparison, a.out)
    elif a.op == "prepare": r = prepare(a.root, a.bindings, a.runtime, a.out)
    elif a.op == "dispatch": r = dispatch(a.registry, submit=a.submit, max_jobs=a.max_jobs)
    elif a.op == "link": r = link_block(a.contract, a.native_plan, a.ledger, a.bindings, a.out)
    else: r = collect(a.root, a.index, a.out)
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
