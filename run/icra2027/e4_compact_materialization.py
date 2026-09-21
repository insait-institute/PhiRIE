"""Source-bound construction-only handoff from the fixed compact E3 pilot.

The existing materializer owns scene validation and A0/A4 artifact export.
This wrapper binds the new execution contract to the complete two-scene source;
it does not require evaluation or an E3 aggregate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import yaml

from robo.eval import e3_factory_materializer as materializer
from robo.eval import agentic_ablation as e3
from robo.manifest.hash import canonical_hash

CODE = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "e2185a8809e9bd3462b3c0449a5367cdf95d18b4"
SOURCE_FREEZE = "20260905-61d8ba4-v1"
SOURCE_CONTRACT = "f80aabb0ee4326b5be667dbd82802917263a5f14f574bd70908ec4559aedfb2a"
DEFAULT_SOURCE = Path("/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027") / SOURCE_FREEZE / "agentic"
SCENES = {"27dd4da69e": 7, "40aec5fffa": 10}
FIXED_TASKS = ["27dd4da69e__obj_1001_to_region", "27dd4da69e__obj_1001_to_obj_1000",
               "40aec5fffa__obj_1001_to_region", "40aec5fffa__obj_1001_to_obj_1005"]


def read(path):
    return yaml.safe_load(Path(path).read_text())


def identity(path):
    path = Path(os.path.abspath(path))
    e3._reject_symlink_components(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing compact source: {path}")
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=e3.sha256_file(path))


def contract_digest(contract):
    expected = canonical_hash({k: v for k, v in contract.items()
                               if k not in {"created_utc", "environment", "contract_sha256"}})
    if contract.get("contract_sha256") != expected:
        raise ValueError("E0 contract digest changed")
    return expected


def resource(contract, name):
    rows = [r for r in contract["resource_inventory"] if r["id"] == name]
    if len(rows) != 1:
        raise ValueError(f"E0 resource missing or duplicated: {name}")
    row = rows[0]
    observed = identity(row["resolved_path"])
    if observed["sha256"] != row["sha256"] or observed["size_bytes"] != row["size_bytes"]:
        raise ValueError(f"E0 resource changed: {name}")
    return observed


def validate_protocol(protocol):
    if (protocol.get("study_scope") != "e4_compact_canonical_engineering"
            or protocol.get("paper_ready") is not False
            or protocol.get("population") != {"scene_ids": list(SCENES), "planned_objects": 17,
                                               "planned_policy_object_rows": 85}
            or [t["task_id"] for t in protocol["fixed_manipulation_tasks"]] != FIXED_TASKS
            or protocol.get("no_substitution_after_qualification") is not True
            or protocol.get("no_policy_outcome_selection") is not True
            or protocol.get("planned_manipulation_episodes") != 40
            or protocol.get("episodes_per_task_arm") != 5
            or protocol.get("construction_arms") != ["A0", "A4"]):
        raise ValueError("fixed compact protocol or denominator differs")


def inspect_source(source_root=DEFAULT_SOURCE):
    """Read-only authentic source review, including typed unfinished dependency."""
    source_root = e3.checked_repo_path(source_root, "compact E3 source", kind="dir")
    source_e0_path = source_root.parent / "contract/freeze_manifest.json"
    source_e0 = read(source_e0_path)
    if (source_root.parent.name != SOURCE_FREEZE or source_e0.get("freeze_id") != SOURCE_FREEZE
            or source_e0["code"].get("commit") != SOURCE_COMMIT
            or source_e0["code"].get("dirty") is not False
            or contract_digest(source_e0) != SOURCE_CONTRACT):
        raise ValueError("compact source E0 identity differs")
    protocol_identity = resource(source_e0, "canonical_scene_roster")
    protocol = read(protocol_identity["path"])
    validate_protocol(protocol)
    jobs_identity = resource(source_e0, "agentic_fresh_jobs")
    source_config = read(jobs_identity["path"])
    jobs, _, audit, inventory_identities = materializer._verify_inventory(source_root)
    if (jobs["freeze_id"] != SOURCE_FREEZE or jobs["source_contract"]["code_commit"] != SOURCE_COMMIT
            or jobs["source_contract"]["contract_sha256"] != SOURCE_CONTRACT
            or jobs["source_contract"]["jobs_sha256"] != canonical_hash(source_config)
            or jobs["counts"] != {"scenes": 2, "jobs": 17, "policy_object_rows": 85}
            or [s["scene_id"] for s in jobs["scenes"]] != list(SCENES)
            or [len(s["jobs"]) for s in jobs["scenes"]] != list(SCENES.values())):
        raise ValueError("compact full inventory identity or population differs")
    sources = source_config["automatic_sources"]
    if [s["scene_id"] for s in sources] != list(SCENES):
        raise ValueError("compact descriptor source scene roster differs")
    waiting, scene_sources = [], {}
    for source in sources:
        scene = source["scene_id"]
        # Authenticate every scene audit, including scenes not selected by a later phase.
        materializer._automatic_scene_audit(jobs, audit, scene)
        missing = [phase for phase in ("observations", "control") if not (source_root / phase / scene).exists()]
        if missing:
            waiting.extend({"scene_id": scene, "dependency": phase,
                            "expected_path": str(source_root / phase / scene)} for phase in missing)
            continue
        receipts = {}
        for phase in ("observe", "control"):
            receipt_root = source_root.parent / "execution_receipts" / scene
            start_path = receipt_root / (phase + "_execution_start.json")
            result_path = receipt_root / (phase + "_execution_result.json")
            if not result_path.exists():
                waiting.append(dict(scene_id=scene, dependency=phase + "_execution_result",
                                    expected_path=str(result_path)))
                continue
            start, result = read(start_path), read(result_path)
            if (start.get("code_commit") != SOURCE_COMMIT
                    or start.get("contract_sha256") != SOURCE_CONTRACT
                    or start.get("scene_id") != scene or start.get("phase") != phase
                    or start.get("planned_jobs") != SCENES[scene]
                    or start.get("planned_policy_object_rows") != 5 * SCENES[scene]
                    or result.get("status") != "PASS" or result.get("exit_code") != 0):
                raise ValueError("compact source execution receipt differs")
            receipts[phase] = dict(start=identity(start_path), result=identity(result_path))
        # A published but malformed directory is an error, never an unfinished job.
        control, shard, _ = e3._load_control_scene(source_root, scene)
        _, observation, points = e3._load_observation_scene(source_root, scene)
        planned = e3._scene_inventory(jobs, scene)
        if (shard["freeze_id"] != SOURCE_FREEZE or shard["job_count"] != SCENES[scene]
                or shard["ledger_row_count"] != 5 * SCENES[scene]
                or set(points) != {j["job_id"] for j in planned["jobs"]}
                or observation.get("code_commit") != SOURCE_COMMIT
                or observation.get("freeze_id") != SOURCE_FREEZE
                or observation.get("evaluation_geometry_read") is not False
                or observation.get("source_scene_gaussian") != planned["source_scene_gaussian"]
                or observation.get("camera_artifacts") != planned["camera_artifacts"]
                or observation.get("observation_protocol") != jobs["observation_protocol"]
                or shard["observation_manifest_sha256"] != e3.sha256_file(source_root / "observations" / scene / "manifest.json")):
            raise ValueError("compact observation/controller identity or denominator differs")
        scene_sources[scene] = dict(
            descriptor={"schema_version": 1, **{k: source[k] for k in
                        ("scene_id", "discovery_directory", "discovery_hashes")}},
            control_seal=identity(control / "seal.json"),
            execution_receipts=receipts,
            observation_seal=identity(source_root / "observations" / scene / "seal.json"),
            observation_manifest=identity(source_root / "observations" / scene / "manifest.json"))
    common = dict(paper_ready=False, planned_objects=17, planned_policy_object_rows=85,
                  planned_manipulation_episodes=40, fixed_task_ids=FIXED_TASKS)
    if waiting:
        return dict(status="WAITING", dependencies=waiting, **common)
    return dict(status="READY", **common, source=dict(
        e3_root=str(source_root), source_e0=identity(source_e0_path),
        protocol=protocol_identity, jobs_config=jobs_identity, inventory=inventory_identities,
        scenes=scene_sources))


def validate_stage(config_path, stage_root):
    config_path = Path(identity(config_path)["path"])
    stage_root = e3.checked_repo_path(stage_root, "compact stage freeze", kind="dir")
    config = read(config_path)
    snapshot = materializer._require_clean_code_snapshot()
    contract = read(stage_root / "contract/freeze_manifest.json")
    contract_digest(contract)
    if (config.get("schema_version") != 1 or config.get("scope") != "e4_compact_materialization_engineering"
            or config.get("paper_ready") is not False or config.get("freeze_id") != stage_root.name
            or contract.get("freeze_id") != stage_root.name
            or contract["code"].get("commit") != snapshot["commit"]
            or contract["code"].get("dirty") is not False
            or materializer.CODE_ROOT != CODE or e3.CODE_ROOT != CODE
            or stage_root.name == SOURCE_FREEZE):
        raise ValueError("compact stage scope or exact source E0 differs")
    if resource(contract, "e4_compact_materialization_config") != identity(config_path):
        raise ValueError("compact stage config differs from E0")
    runtime = resource(contract, "materialization_python")
    if Path(runtime["path"]).resolve() != Path(config["python"]).resolve() or Path(runtime["path"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("compact materialization interpreter differs")
    reviewed = inspect_source(config["e3_root"])
    if reviewed["status"] == "READY" and reviewed["source"] != config["source"]:
        raise ValueError("compact frozen source bundle changed")
    return config, reviewed, snapshot


def run(config_path, stage_root, *, scene_id=None, handoff=False):
    started = time.monotonic()
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise ValueError("ordinary individual jobs required")
    if os.environ.get("SLURM_GPUS_ON_NODE", "0") not in {"", "0"}:
        raise ValueError("compact materialization must request CPU resources only")
    if handoff == (scene_id is not None) or (scene_id is not None and scene_id not in SCENES):
        raise ValueError("choose one declared scene or the final handoff")
    config, reviewed, snapshot = validate_stage(config_path, stage_root)
    if reviewed["status"] == "WAITING":
        return reviewed  # no output directory or fabricated artifact failures
    stage_root = Path(stage_root).resolve()
    directory = stage_root / "materialization"
    if handoff:
        result = dict(schema_version=1, status="PASS", paper_ready=False,
                      source=config["source"], materializer_commit=snapshot["commit"],
                      fixed_task_ids=FIXED_TASKS, planned_manipulation_episodes=40, scenes={})
        for scene in SCENES:
            receipt = directory / scene / "result.json"
            if not receipt.is_file():
                return dict(status="WAITING", paper_ready=False, dependency=str(receipt))
            recorded = read(receipt)
            if (recorded.get("source") != config["source"] or recorded.get("scene_id") != scene
                    or recorded.get("materializer_commit") != snapshot["commit"]
                    or recorded.get("status") != "PASS" or recorded.get("paper_ready") is not False
                    or recorded.get("planned_objects") != SCENES[scene]
                    or recorded.get("planned_manipulation_episodes") != 20):
                raise ValueError("compact scene receipt differs from frozen source")
            for policy in ("A0", "A4"):
                destination = directory / scene / policy
                validation = materializer.validate_materialized_factory(destination, expected_scene_id=scene, expected_policy_id=policy)
                expected = dict(validation=validation, manifest=identity(destination / "materialization_manifest.json"))
                if recorded["variants"].get(policy) != expected:
                    raise ValueError("compact scene receipt variant identity changed")
            result["scenes"][scene] = dict(receipt=identity(receipt),
                descriptor=identity(directory / scene / "automatic_scene_descriptor.json"),
                variants={p: identity(directory / scene / p / "materialization_manifest.json") for p in ("A0", "A4")})
        materializer._write_inside(directory / "handoff.json", materializer._json_bytes(result))
        return result
    scene_root = directory / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)
    report_path = scene_root / "result.json"
    if report_path.exists():
        raise FileExistsError("completed compact scene may not be overwritten")
    descriptor = scene_root / "automatic_scene_descriptor.json"
    payload = materializer._json_bytes(config["source"]["scenes"][scene_id]["descriptor"])
    if descriptor.exists():
        if descriptor.is_symlink() or descriptor.read_bytes() != payload:
            raise ValueError("compact published descriptor changed")
    else:
        materializer._write_inside(descriptor, payload)
    report = dict(schema_version=1, scene_id=scene_id, status="PASS", paper_ready=False,
                  source=config["source"], materializer_commit=snapshot["commit"],
                  stage_freeze_id=stage_root.name, stage_config=identity(config_path),
                  execution=dict(slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      node=socket.gethostname(), gpu_requested=False,
                      cpus=os.environ.get("SLURM_CPUS_PER_TASK"), python=sys.executable),
                  planned_objects=SCENES[scene_id], planned_manipulation_episodes=20, variants={})
    for policy in ("A0", "A4"):
        destination = scene_root / policy
        if not destination.exists():
            materializer.materialize_factory_variant(e3_root=config["e3_root"], scene_id=scene_id,
                policy_id=policy, out=destination, automatic_scene_contract=descriptor)
        validated = materializer.validate_materialized_factory(destination, expected_scene_id=scene_id, expected_policy_id=policy)
        manifest = read(destination / "materialization_manifest.json")
        if (manifest["e3_freeze_id"] != SOURCE_FREEZE or manifest["e3_code_commit"] != SOURCE_COMMIT
                or manifest["source_scene"]["automatic_scene_descriptor"] != materializer._identity(descriptor)
                or validated["roster"]["job_count"] != SCENES[scene_id]):
            raise ValueError("existing compact variant belongs to another frozen source")
        report["variants"][policy] = dict(validation=validated, manifest=identity(destination / "materialization_manifest.json"))
    report["wall_s"] = time.monotonic() - started
    materializer._write_inside(report_path, materializer._json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--config")
    parser.add_argument("--freeze-root")
    parser.add_argument("--scene-id", choices=list(SCENES))
    parser.add_argument("--handoff", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        result = inspect_source(args.source_root)
    else:
        if not args.config or not args.freeze_root:
            parser.error("execution requires an E0-bound config and new freeze root")
        result = run(args.config, args.freeze_root, scene_id=args.scene_id, handoff=args.handoff)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
