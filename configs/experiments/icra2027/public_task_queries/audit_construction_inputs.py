"""Audit E6 input readiness without fabricating construction bundles/features.

Only reads this package, the declared public source files, and the named
integration config/code files. Never scans outputs or follows manifest refs.
Exit 3 means prospective population retained but construction inputs missing
or unreviewed. Exit 2 means malformed inputs. This audit does not publish features.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from validate import CONDITIONS, HERE, SCENES, digest, require, validate

CONTRACT_CODE = [
    "robo/eval/build_task_support_dataset.py",
    "robo/certification/task_graph.py",
    "robo/certification/grounding.py",
    "robo/certification/features/visual.py",
    "robo/certification/features/geometry.py",
    "robo/certification/features/support_contact.py",
    "robo/certification/features/robot_control.py",
]
REQUIRED_INPUTS = {
    "resolved_freeze_id": "A concrete common freeze_id instead of inherit.",
    "registered_public_roots": "Explicitly audited public construction roots; no legacy/oracle/GT/evaluation paths.",
    "hashed_population_manifest": "A hashed manifest with freeze_id and exactly 72 rows keyed by freeze_id, scene_id, task_id, condition_id.",
    "terminal_build_provenance": "Per scene-condition terminal outcome from a real construction attempt. Do not call an unattempted/missing build failed.",
    "metric_scene": "For complete builds: scene_id, objects with construction-local id/label/metric aabb, robot.base_pos, and any declared policy cameras with id/pos/look_at in the same validated frame.",
    "query_to_construction_association": "Per query-condition public association from the immutable reference RGB roles/regions to construction entities; missing or ambiguous roles stay unresolved.",
    "target_region_geometry": "Distinct metric placement patches or receptacle interiors grounded from public construction evidence; a shared tabletop entity alone cannot distinguish its different target patches.",
    "policy_camera_and_robot_frame": "Public metric robot alignment and policy-camera identity/view specification; capture intrinsics/poses alone do not establish these.",
    "frame_state_consistency": "Public evidence that the construction state supports the query's reference-frame instances, without assuming consistency across capture times.",
}
OPTIONAL_EVIDENCE = {
    "visual": ["objects[].checks.removal_alpha_coverage.alpha_cov", "objects[].checks.removal_depth_residual.depth_plane_mm", "objects[].checks.removal_redetection.det_after", "objects[].checks.ghost_object.status"],
    "geometry": ["objects[].checks.registration_residual.chamfer_med_mm", "objects[].checks.registration_residual.icp_tilt_deg", "objects[].checks.candidate_disagreement.spread_mm", "objects[].checks.scale_sanity.size_ratio_vs_obs", "objects[].checks.visual_collision_surface_distance.p95_distance_mm"],
    "support": ["objects[].checks.penetration.status", "objects[].checks.settle_drift.drift_mm", "objects[].checks.support_overlap.status", "objects[].checks.drop_response.status"],
}


def compose_report(sources, roster, expansion, config, input_hashes):
    require(config["scene_ids"] == SCENES and config["conditions"] == CONDITIONS,
            "integration population differs from immutable query roster")
    require(len(roster["queries"]) == 24 and len(expansion["rows"]) == 72,
            "prospective query population incomplete")
    queries = {(q["scene_id"], q["task_id"]): q for q in roster["queries"]}
    require(len(queries) == 24, "duplicate canonical query")
    for q in queries.values():
        require(q["query_sha256"] == digest({k:v for k,v in q.items() if k != "query_sha256"}), "canonical task changed")
    available = {s["scene_id"]: s for s in sources["scenes"]}
    planned = []
    for row in expansion["rows"]:
        q = queries[(row["scene_id"], row["task_id"])]
        require(row["query_sha256"] == q["query_sha256"], "condition task changed")
        planned.append({
            "scene_id": row["scene_id"], "task_id": row["task_id"],
            "condition_id": row["condition_id"], "query_sha256": row["query_sha256"],
            "query_ref": row["query_ref"], "task_family": q["task_family"],
            "reference_image_id": q["source_state_anchor"],
            "construction_input_status": "missing_registered_manifest" if config.get("public_manifest") is None else "declared_manifest_not_inspected",
            "terminal_build_status": None,
            "terminal_build_status_reason": "No admissible terminal construction evidence inspected; absence is not an observed failed build.",
            "retain_in_planned_denominator": True,
            "missing_requirement_codes": list(REQUIRED_INPUTS),
        })
    expected = {(s,t,c) for s,t in queries for c in CONDITIONS}
    actual = {(r["scene_id"],r["task_id"],r["condition_id"]) for r in planned}
    require(actual == expected and len(actual) == 72, "condition population changed")
    counts = Counter((r["scene_id"],r["condition_id"]) for r in planned)
    require(all(v == 4 for v in counts.values()) and len(counts) == 18,
            "scene-condition query coverage changed")
    return {
        "schema_version": 1, "kind": "prospective_construction_missing_input_audit",
        "status": "BLOCKED_MISSING_OR_UNREVIEWED_CONSTRUCTION_INPUTS", "paper_ready": False,
        "features_ready": False, "feature_rows_written": 0, "construction_bundles_written": 0,
        "scope": "Only named config/code and declared public RGB/camera files; no output directory scan and no private/evaluation input reads. No construction existence claim outside registered inputs.",
        "input_hashes": input_hashes,
        "registered_configuration": {
            "freeze_id": config.get("freeze_id"), "public_roots": config.get("public_roots"),
            "public_manifest": config.get("public_manifest"),
            "task_roster_is_populated": all(config.get("task_ids_by_scene",{}).get(s) for s in SCENES),
            "task_roster_overlay_available": "audit_roster_overlay.json",
        },
        "population": {"scenes": 6, "queries": 24, "conditions": 3, "prospective_rows": 72,
                       "scene_condition_construction_states": 18, "actual_terminal_build_counts": None,
                       "unattempted_rows_not_reclassified_as_failed": True},
        "available_public_inputs": [{"scene_id":s,"public_rgb_count":available[s]["public_rgb_count"],
                                     "inspected_rgb_count":5,"hashed_camera_source_count":2,
                                     "source_manifest_ref":"sources.json#"+s,
                                     "canonical_query_count":4} for s in SCENES],
        "required_inputs": REQUIRED_INPUTS,
        "optional_evidence_fields": OPTIONAL_EVIDENCE,
        "optional_evidence_missingness": "Missing checks must remain None/empty with indicators. Missing measurements do not justify fabricated favorable values or fabricated build failures.",
        "feature_stage_bundle_contract": {
            "population_manifest": {"required_fields":["freeze_id","rows"],
                                    "row_fields":["freeze_id","scene_id","task_id","condition_id","task_family","build_manifest"],
                                    "build_manifest_reference_fields":["path","sha256"]},
            "bundle_required_fields":["freeze_id","scene_id","task_id","condition_id","evidence_source","source_kind","build_status"],
            "evidence_source":"construction_observation", "source_kind":"real",
            "supported_terminal_statuses":["complete","failed","rejected","abstained"],
            "complete_bundle_additions":["scene","task"], "optional_audit_field":"audit",
            "task_content_required":["task_id","roles","language.instruction"],
            "adapter_additional_provenance_required":["query_sha256","source_state_anchor","public_role_association_provenance"],
            "failed_bundle_policy":"Preserve its planned row with actual terminal provenance; driver produces explicit build-failed graph and missing evidence. Do not convert missing inputs into failed bundles.",
        },
        "interface_limitations":[
            "Current graph builder does not consume image boxes/polygons. It needs construction metric entities; no image-to-3D adapter is implemented here.",
            "Distinct destination patches on one support must not collapse into one undifferentiated target object.",
            "Current swept-workspace proxy uses robot-to-manipulated-object anchor when resolved; it does not certify the full object-to-destination transport segment.",
            "Public feature payload filter prohibits hidden/evaluation fields and role_refs. Public construction IDs must never be sourced from a hidden identity mapping.",
        ],
        "next_action":"Supply a specifically audited public construction root/manifest with real terminal outcomes and metric role/region evidence, then implement a content-preserving association adapter. Do not run the feature stage on these request rows.",
        "rows": planned,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-root", type=Path,
                        default=Path("/group/worldcept/PhiRIE/code/SimAny-wt/icra-integration"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        names = ["sources.json", "queries.json", "condition_expansion.json"]
        package = [json.loads((HERE/name).read_text()) for name in names]
        validate(*package, verify_source_bytes=True)
        config_path = args.integration_root / "configs/experiments/icra2027/audit.yaml"
        config = yaml.safe_load(config_path.read_text())
        files = [*(HERE/name for name in names), config_path,
                 *(args.integration_root/name for name in CONTRACT_CODE)]
        hashes = [{"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
        report = compose_report(*package, config, hashes)
        # Real negative checks: population loss, condition-content tampering,
        # and incompatible scene roster must be rejected before any output.
        mutations = []
        e = copy.deepcopy(package[2]); e["rows"].pop(); mutations.append((package[0],package[1],e,config))
        e = copy.deepcopy(package[2]); e["rows"][0]["query_sha256"] = "0"*64; mutations.append((package[0],package[1],e,config))
        c = copy.deepcopy(config); c["scene_ids"] = SCENES[:-1]; mutations.append((*package,c))
        for values in mutations:
            try:
                compose_report(*values, hashes)
            except ValueError:
                continue
            raise AssertionError("readiness audit accepted changed population/content")
        report["negative_tests_passed"] = len(mutations)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(report,handle,indent=2,sort_keys=True); handle.write("\n")
        print(json.dumps({"out":str(args.out),"status":report["status"],"retained_prospective_rows":72,"feature_rows_written":0,"negative_tests_passed":len(mutations)}))
        return 3
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
