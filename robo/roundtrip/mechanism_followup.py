"""Fresh same-process B0/B1/B2/B3 comparisons on an already sealed roster.

This is a mechanism follow-up on an observed TEST cohort, NOT a new independent
confirmatory test. No generation, task selection, or threshold tuning occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shlex
import subprocess

METHODS = ("REF_NATIVE", "B0_FIXED_NATIVE", "B1_FIXED_PRIORITY", "B2_EVIDENCE", "B3_AGENT_NATIVE")
PROTOCOL = "native_mechanism_followup_v1"


def _core():
    from robo.roundtrip import matrix
    return matrix


def expected_engine_methods(selected, pilot=False):
    """Admit only the exact new block; preserve all legacy engine contracts."""
    legacy = ["REF_NATIVE", "B0_FIXED_NATIVE", "B3_AGENT_NATIVE",
              "B4_ROOM_REPAIR_NATIVE", "BM_BUDGET_MATCHED_NATIVE"]
    protocols = {json.loads(Path(u["config_path"]).read_text()).get("mechanism_followup")
                 for u in selected}
    if protocols == {None}:
        return legacy[:2] if pilot else legacy
    if pilot or protocols != {PROTOCOL}:
        raise ValueError("mixed or unsupported mechanism follow-up engine protocol")
    if any(u["scope"] != "L0_target_only" or u["renderer"] != "native"
           or u["execution_protocol"] != "primary_native" for u in selected):
        raise ValueError("mechanism engine supports only primary native L0")
    return list(METHODS)


def worker_identity(root):
    root = Path(root).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError("worker checkout has tracked uncommitted changes")
    return {"root": str(root), "commit": commit}


def native_asset_closure(binding):
    """Seal immutable imported inputs, not a path-only or geometry-only identity."""
    m = _core()
    if binding.get("accepted") is not True:
        return {}
    root = Path(binding["object_dir"]).resolve(strict=True)
    required = {"aligned.json", "physics.json", "mesh_sim.obj"}
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    names = {str(p.relative_to(root)) for p in paths}
    if not required <= names or not any(n.startswith("collision/part_") for n in names):
        raise ValueError("accepted native asset is incomplete")
    if any(p.is_symlink() for p in paths):
        raise ValueError("native asset closure must not contain symlinks")
    return {str(p): m.sha(p) for p in paths}


def invalid_collision_parts(directory):
    """The unchanged native import predicate, not a confidence/stability filter."""
    import trimesh
    invalid = []
    for path in sorted((Path(directory) / "collision").glob("part_*.obj")):
        mesh = trimesh.load(path, process=False, force="mesh")
        if not (mesh.is_watertight and mesh.is_convex and mesh.volume > 0):
            invalid.append(str(path.resolve()))
    return invalid


def read_config(path):
    import yaml
    path = Path(path).resolve(strict=True)
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("requires native_mechanism_followup_v1 schema 1")
    if config.get("execution_ready") is not True:
        raise ValueError("resolve paths and set execution_ready=true after preflight")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.get("cohort_id", "")):
        raise ValueError("declare a new cohort_id")
    if tuple(config.get("methods", METHODS)) != METHODS:
        raise ValueError("the complete ordered REF/B0/B1/B2/B3 block is required")
    for key in ("source_planned_units", "source_bindings", "engine_admission", "native_python", "worker_source"):
        p = Path(config[key]).expanduser()
        config[key] = str((path.parent / p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True))
    for key in ("source_build_bindings", "candidate_pool_roots"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"nonempty {key} list required")
        config[key] = [str((path.parent / Path(p)).resolve(strict=True)) for p in config[key]]
    config["source_config"] = str(path)
    return config


def pool_index(roots):
    """Index target pools only, rejecting contradictory duplicate snapshots."""
    m = _core(); index = {}
    for root in roots:
        root = Path(root)
        paths = [root] if root.is_file() else sorted(root.rglob("candidate_pool.json"))
        for path in paths:
            pool = json.loads(path.read_text())
            if pool.get("object_role", "target") != "target":
                continue
            iid = pool.get("canonical_instance_id")
            if not iid:
                continue
            if iid in index and m.sha(index[iid]) != m.sha(path):
                raise ValueError(f"multiple different target pools for {iid}; specify exact roots")
            index[iid] = path.resolve()
    return index


def pool_binding(path, method, capture_hash):
    """Use the already executed A1/A2 decision and its exact native asset."""
    m = _core(); path = Path(path).resolve(strict=True)
    pool = json.loads(path.read_text())
    if pool.get("schema_version") != 2 or pool.get("capture_manifest_sha256") != capture_hash:
        raise ValueError("candidate pool/capture mismatch")
    rows = [r for r in pool["outcomes"] if r.get("native_method") == method]
    if len(rows) != 1:
        raise ValueError(f"pool needs exactly one {method} outcome")
    row = rows[0]; pid = row.get("selected_proposal_id")
    if method not in METHODS[2:4]:
        raise ValueError("pool_binding is only for frozen B1/B2 initial selections")
    initial_ids = {c["proposal_id"] for c in pool["initial_candidates"]}
    if pid not in initial_ids:
        raise ValueError("B1/B2 must select initial proposals, never retry artifacts")
    obj = path.parent / Path(row["object_dir"]).name
    hashes = row.get("artifact_hashes", {})
    if not {"aligned.json", "physics.json", "mesh_sim.obj"}.issubset(hashes):
        raise ValueError("selected asset lacks a complete native identity")
    if not any(k.startswith("collision/") for k in hashes):
        raise ValueError("selected asset lacks collision parts")
    for rel, digest in hashes.items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or (obj / p).is_symlink() or m.sha(obj / p) != digest:
            raise ValueError(f"candidate artifact changed: {rel}")
    return dict(canonical_instance_id=pool["canonical_instance_id"], controller_method=method,
                accepted=True, terminal_status="READY", object_dir=str(obj.resolve()),
                build_manifest=str(path), build_manifest_sha256=m.sha(path),
                selected_proposal_id=pid, asset_hashes=hashes,
                decision_source="frozen candidate_pool.outcomes; no reselection or regeneration")


def followup_rows(source, cohort_id):
    """Preserve every canonical/reset identity, but never reuse old REF results."""
    m = _core()
    refs = [u for u in source if u["controller_method"] == "REF_NATIVE"]
    if not refs or cohort_id in {u["cohort_id"] for u in source}:
        raise ValueError("nonempty reference roster and a NEW cohort_id required")
    keys = [(u["canonical_instance_id"], u["reset_id"]) for u in refs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate canonical/reset reference")
    if any(u["scope"] != "L0_target_only" or u["renderer"] != "native" for u in refs):
        raise ValueError("follow-up currently supports only the declared L0 native scope")
    rows = []
    for ref in refs:
        for method in METHODS:
            u = copy.deepcopy(ref)
            for key in ("config_path", "config_sha256", "result_path_planned", "result", "result_path",
                        "result_sha256", "argv", "source_code", "native_config_source", "native_config_source_sha256"):
                u.pop(key, None)
            u.update(cohort_id=cohort_id, controller_method=method, terminal_status="NOT_SCHEDULED",
                     executed=None, success=None)
            identity = {k: u[k] for k in m.IDENTITY_FIELDS}
            u["unit_id"] = "unit-" + m.canonical_hash(identity)[:32]
            u["plan_unit_id"] = "plan-" + m.canonical_hash(identity)[:24]
            rows.append(u)
    return rows


def prepare(config, out):
    m = _core()
    from robo.roundtrip.spec import validate_spec
    from robo.roundtrip.cohort_dispatch import nonrollout
    source = m.read_rows(config["source_planned_units"])
    rows = followup_rows(source, config["cohort_id"])
    refs = {(u["canonical_instance_id"], u["reset_id"]): u for u in source if u["controller_method"] == "REF_NATIVE"}
    bindings = {b["canonical_instance_id"]: b for b in m.read_rows(config["source_bindings"])}
    pools = pool_index(config["candidate_pool_roots"])
    old = {}
    for path in config["source_build_bindings"]:
        paths = [Path(path)] if Path(path).is_file() else sorted(Path(path).rglob("build_bindings.jsonl"))
        for p in paths:
            for b in m.read_rows(p):
                key = (b["canonical_instance_id"], b["controller_method"])
                if key[1] not in ("B0_FIXED_NATIVE", "B3_AGENT_NATIVE"):
                    continue
                if key in old and m.canonical_hash(old[key]) != m.canonical_hash(b):
                    raise ValueError(f"conflicting source build binding: {key}")
                if m.sha(b["build_manifest"]) != b["build_manifest_sha256"]:
                    raise ValueError("source build manifest changed")
                old[key] = b
    builds = []; input_hashes = {config["source_planned_units"]: m.sha(config["source_planned_units"]),
                               config["source_bindings"]: m.sha(config["source_bindings"]),
                               config["engine_admission"]: m.sha(config["engine_admission"])}
    for iid in sorted({u["canonical_instance_id"] for u in rows}):
        if iid not in bindings:
            raise ValueError(f"missing canonical acquisition binding: {iid}")
        b0 = old.get((iid, "B0_FIXED_NATIVE")); b3 = old.get((iid, "B3_AGENT_NATIVE"))
        if b0 is None or b3 is None:
            raise ValueError(f"missing frozen B0/B3 binding: {iid}")
        builds.extend([b0, b3])
        for b in (b0, b3):
            input_hashes[b["build_manifest"]] = b["build_manifest_sha256"]
        if iid in pools:
            for method in METHODS[2:4]:
                builds.append(pool_binding(pools[iid], method, bindings[iid]["capture_manifest_sha256"]))
            if json.loads(pools[iid].read_text())["canonical_instance_id"] != iid:
                raise ValueError("candidate pool canonical identity differs")
            input_hashes[str(pools[iid])] = m.sha(pools[iid])
        else:
            shared_rule = "all arms require the same frozen automatic TRAIN discovery; no initial pool exists"
            if not (b0.get("accepted") is False and b3.get("accepted") is False
                    and b0.get("terminal_status") == b3.get("terminal_status") == "BUILD_FAILED"
                    and b3.get("propagation_rule") == shared_rule):
                raise ValueError(f"missing pool is NOT automatically a method failure: {iid}")
            for evidence_path, digest in b3.get("source_evidence", {}).items():
                if m.sha(evidence_path) != digest:
                    raise ValueError("shared discovery failure evidence changed")
                input_hashes[evidence_path] = digest
            for method in METHODS[2:4]:
                builds.append(dict(b3, controller_method=method, propagation_rule=b3["propagation_rule"]))
    for build in builds:
        input_hashes.update(native_asset_closure(build))
    identity = worker_identity(config["worker_source"])
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    # Classify supplied invalid collision assets before invoking the policy.
    # This is the original native importer predicate, with no threshold changes.
    validity_cache = {}
    for i, build in enumerate(builds):
        if build.get("accepted") is not True:
            continue
        directory = build["object_dir"]
        if directory not in validity_cache:
            validity_cache[directory] = invalid_collision_parts(directory)
        invalid = validity_cache[directory]
        if invalid:
            receipt_path = out / "import_validity" / (build["canonical_instance_id"] + "-" + build["controller_method"] + ".json")
            evidence = {p: input_hashes[p] for p in invalid}
            m.save_new(receipt_path, {"parent_binding": build, "invalid_collision_parts": evidence,
                "predicate": "is_watertight and is_convex and volume>0", "original_geometry_changed": False})
            builds[i] = dict(build, accepted=False, terminal_status="BUILD_FAILED", object_dir=None,
                build_manifest=str(receipt_path), build_manifest_sha256=m.sha(receipt_path),
                failure="original supplied collision fails unchanged native import predicate", source_evidence=evidence)
    workers = out / "workers"; emitted = []
    for u in rows:
        ref = refs[(u["canonical_instance_id"], u["reset_id"])]
        c = json.loads(Path(ref["config_path"]).read_text())
        if m.canonical_hash(c) != ref["config_sha256"]:
            raise ValueError("source reference config changed")
        c.update(cohort_id=config["cohort_id"], controller_method=u["controller_method"],
                 mechanism_followup=PROTOCOL, policy_engine_protocol="per_canonical_engine_v1")
        validate_spec(c)
        p = out / "configs" / (u["unit_id"] + ".json"); m.save_new(p, c)
        emitted.append(dict(u, config_path=str(p), config_sha256=m.canonical_hash(c),
                            result_path_planned=str(workers / u["unit_id"] / "runner/episode/result.json")))
        input_hashes[ref["config_path"]] = m.sha(ref["config_path"])
    commands = m.commands_for_units(emitted, builds, python_native=config["native_python"])
    by_build = {(b["canonical_instance_id"], b["controller_method"]): b for b in builds}
    for u in emitted:
        b = by_build.get((u["canonical_instance_id"], u["controller_method"]))
        if b and b.get("terminal_status") in ("BUILD_FAILED", "ABSTAINED"):
            m.save_new(workers / u["unit_id"] / "terminal.json", nonrollout(u, b))
    m.save_new(out / "planned_units.jsonl", emitted, jsonl=True)
    m.save_new(out / "commands.json", commands); m.save_new(out / "build_bindings.jsonl", builds, jsonl=True)
    m.save_new(out / "plan.json", dict(protocol=PROTOCOL, config=config, input_hashes=input_hashes,
        planned_units_sha256=m.sha(out / "planned_units.jsonl"), commands_sha256=m.sha(out / "commands.json"),
        planned=len(emitted), instances=len({u["canonical_instance_id"] for u in rows}),
        worker_identity=identity, methods=list(METHODS), old_outcomes_reused=False,
        interpretation="post-publication mechanism follow-up on existing roster; not an independent TEST confirmation"))
    return emitted


def launch(bundle, *, submit=False, max_jobs=1):
    m = _core(); bundle = Path(bundle).resolve(strict=True)
    plan = json.loads((bundle / "plan.json").read_text()); config = plan["config"]
    if worker_identity(config["worker_source"]) != plan["worker_identity"]:
        raise ValueError("worker checkout changed after plan sealing; prepare a new bundle")
    for path, digest in plan["input_hashes"].items():
        if m.sha(path) != digest:
            raise ValueError("frozen input changed: " + path)
    for name in ("planned_units", "commands"):
        p = bundle / (name + (".jsonl" if name == "planned_units" else ".json"))
        if m.sha(p) != plan[name + "_sha256"]:
            raise ValueError("prepared plan changed")
    if max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    flags = config.get("sbatch_args", [])
    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        raise ValueError("sbatch_args must be explicit string arguments")
    if any((x.startswith("-a") and not x.startswith("--")) or x.startswith("--array") or x.startswith("--wrap") for x in flags):
        raise ValueError("job arrays and shell-wrap overrides are prohibited")
    units = m.read_rows(bundle / "planned_units.jsonl"); commands = json.loads((bundle / "commands.json").read_text())
    from robo.roundtrip.local_policy_instance import preflight_engine
    report = []
    for iid in sorted({u["canonical_instance_id"] for u in units}):
        job = bundle / "jobs" / iid
        if (job / "submission_intent.json").exists():
            continue
        selected = [u for u in units if u["canonical_instance_id"] == iid]
        n = len([u for u in selected if u["controller_method"] == "REF_NATIVE"])
        preflight_engine(selected, commands, bundle / "workers", list(METHODS), n)
        argv = ["bash", "run/roundtrip/local_policy_instance.sh", "--planned", str(bundle / "planned_units.jsonl"),
                "--commands", str(bundle / "commands.json"), "--instance-id", iid,
                "--worker-root", str(bundle / "workers"), "--out", str(job / "endpoint"),
                "--expected-resets", str(n), "--engine-protocol", "per_canonical_engine_v1",
                "--admission", config["engine_admission"]]
        script = job / "engine.sbatch"
        command = ["sbatch", "--parsable", *flags, "--job-name=mechanism-" + iid[-10:],
                   "--output=" + str(job / "engine-%j.log"), str(script)]
        report.append(dict(instance=iid, planned=len(selected), command=command, submitted=submit))
        if submit:
            job.mkdir(parents=True, exist_ok=True)
            with script.open("x") as f:
                f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(config["worker_source"]) + "\nexec " + shlex.join(argv) + "\n")
            m.save_new(job / "submission_intent.json", report[-1])
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            m.save_new(job / "submission.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                job_id=result.stdout.strip().split(";")[0] if result.returncode == 0 else None))
            if result.returncode:
                raise RuntimeError("sbatch failed; intent retained, inspect submission.json before retry")
        if len(report) >= max_jobs:
            break
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--config", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("launch"); q.add_argument("--bundle", required=True); q.add_argument("--submit", action="store_true"); q.add_argument("--max-jobs", type=int, default=1)
    q = sub.add_parser("collect"); q.add_argument("--bundle", required=True); q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.phase == "prepare":
        result = dict(planned=len(prepare(read_config(a.config), a.out)), out=a.out)
    elif a.phase == "launch":
        result = launch(a.bundle, submit=a.submit, max_jobs=a.max_jobs)
    else:
        m = _core(); root = Path(a.bundle).resolve(strict=True)
        rows = m.collect(m.read_rows(root / "planned_units.jsonl"), root / "workers", a.out)
        output = Path(a.out).resolve()
        m.save_new(output / "paper_pipeline.json", {"native_scale_up": {
            "planned_units": str(root / "planned_units.jsonl"),
            "episode_ledger": str(output / "episode_ledger.jsonl")}})
        result = dict(rows=len(rows), pipeline_config=str(output / "paper_pipeline.json"))
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
