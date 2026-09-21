from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from agents.assets import s5_align
from agents.orchestrator import runtime as orchestrator_runtime
from agents.orchestrator.evidence import REASON_CODES_IN_ORDER
from agents.orchestrator.policies import validate_policy_config
from robo.eval import agentic_ablation


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_PATH = REPO_ROOT / "configs/experiments/icra2027/agentic_jobs.yaml"
POLICIES_PATH = REPO_ROOT / "configs/experiments/icra2027/agentic_policies.yaml"
CONSTRUCTION_PATH = REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _policy_map(config: dict) -> dict[str, dict]:
    return {row["policy_id"]: row for row in config["policies"]}


def _check_map(config: dict) -> dict[str, dict]:
    return {row["field"]: row for row in config["primary_gate"]["checks"]}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def test_agentic_policy_executor_contract_matches_detailed_contract() -> None:
    config = _load(POLICIES_PATH)
    validate_policy_config(config)
    assert config["schema_version"] == 1
    assert config["study_scope"] == "conditional_fixed_gt_factory_jobs"
    assert config["paper_ready"] is False
    assert config["threshold_status"] == "development_preliminary"
    assert config["threshold_selection_population"] == "none"
    assert config["evaluation_population_may_tune_thresholds"] is False

    executor = config["executor_contract"]
    assert config["gates"] == executor["gates"]
    assert config["fixed_priority"] == executor["fixed_priority"]
    assert config["retry"] == executor["retry"]
    assert executor["max_retries"] == 1
    assert executor["retry"]["max_retries"] == 1
    assert executor["retry"]["require_new_transform_hash"] is True
    gates = executor["gates"]
    assert gates == {
        "min_observation_points": 200,
        "max_registration_residual_m": 0.04,
        "min_scale_ratio": 0.4,
        "max_scale_ratio": 2.5,
        "min_visible_fraction": 0.5,
        "max_abs_support_gap_m": 0.015,
        "min_support_overlap_fraction": 0.15,
        "require_collision_valid": True,
        "max_collision_parts": 32,
        "require_drop_stable": True,
        "max_settle_drift_m": 0.03,
        "max_initial_penetration_m": 0.01,
        "selection_tool_order": ["trellis", "reconviagen", "registration_retry"],
    }
    assert config["controller_contract"]["deterministic_tool_tiebreak_order"] == gates[
        "selection_tool_order"
    ]
    assert config["primary_gate"]["reason_codes_in_order"] == list(
        REASON_CODES_IN_ORDER
    )

    checks = _check_map(config)
    assert checks["observation_point_count"]["value"] == gates["min_observation_points"]
    assert checks["symmetric_clipped_registration_residual_m"]["value"] == gates["max_registration_residual_m"]
    assert checks["scale_ratio_vs_observation"]["value"] == [
        gates["min_scale_ratio"],
        gates["max_scale_ratio"],
    ]
    assert checks["visible_fraction"]["value"] == gates["min_visible_fraction"]
    assert checks["absolute_support_gap_m"]["value"] == gates["max_abs_support_gap_m"]
    assert checks["support_overlap_fraction"]["value"] == gates["min_support_overlap_fraction"]
    assert checks["initial_penetration_m"]["value"] == gates["max_initial_penetration_m"]
    assert checks["collision_valid"]["value"] is gates["require_collision_valid"]
    assert checks["usable_convex_parts"]["value"] == [1, gates["max_collision_parts"]]
    assert checks["settle_stable"]["value"] is gates["require_drop_stable"]
    assert checks["settle_drift_m"]["value"] == gates["max_settle_drift_m"]
    assert checks["settle_sunk"] == {
        "field": "settle_sunk", "operator": "equals", "value": False
    }

    raw_fields = config["construction_evidence"]["raw_fields"]
    assert "scale_ratio_vs_observation" in raw_fields
    assert "size_ratio_vs_observation" not in raw_fields
    assert {
        "symmetric_clipped_registration_residual_m",
        "visible_fraction",
        "usable_convex_parts",
        "settle_stable",
        "settle_sunk",
    }.issubset(raw_fields)


def test_agentic_probe_contract_matches_runtime_constants() -> None:
    config = _load(POLICIES_PATH)
    probe = config["executor_contract"]["probe"]
    assert probe == {
        "mesh_sample_seed": 42,
        "mesh_sample_count": 20000,
        "symmetric_clip_distance_m": 0.03,
        "support_band_m": 0.01,
        "initial_support_gap_m": 0.005,
        "collision_backend": "deterministic_single_convex_hull_v1",
        "collision_parts_generated": 1,
        "pybullet_hz": 240,
        "velocity_zero_settle_s": 2,
        "free_settle_s": 2,
        "sunk_when_aabb_min_z_m_less_than": -0.01,
        "settle_stable_drift_threshold_m": 0.03,
        "settle_stable_requires_not_sunk": True,
        "category_independent_physics": {
            "mass_kg": 0.3,
            "friction": 0.5,
            "restitution": 0.1,
        },
    }
    assert probe["mesh_sample_seed"] == orchestrator_runtime.MESH_SAMPLE_SEED
    assert probe["mesh_sample_count"] == orchestrator_runtime.MESH_SAMPLE_COUNT
    assert probe["symmetric_clip_distance_m"] == s5_align.ICP_DIST
    assert probe["support_band_m"] == orchestrator_runtime.SUPPORT_BAND_M
    assert probe["initial_support_gap_m"] == orchestrator_runtime.INITIAL_SUPPORT_GAP_M
    assert probe["pybullet_hz"] == orchestrator_runtime.DROP_HZ
    assert probe["velocity_zero_settle_s"] == orchestrator_runtime.SETTLE_SECONDS
    assert probe["free_settle_s"] == orchestrator_runtime.FREE_SECONDS
    assert (
        probe["sunk_when_aabb_min_z_m_less_than"]
        == orchestrator_runtime.SUNK_AABB_MIN_Z_M
    )
    assert (
        probe["settle_stable_drift_threshold_m"]
        == orchestrator_runtime.STABLE_DRIFT_M
        == config["gates"]["max_settle_drift_m"]
    )
    assert probe["category_independent_physics"] == {
        key: orchestrator_runtime._frozen_physics()[key]
        for key in ("mass_kg", "friction", "restitution")
    }
    assert (
        config["construction_evidence"]["symmetric_residual_definition"]["clip_distance_m"]
        == probe["symmetric_clip_distance_m"]
    )

    drifted = dict(config)
    drifted["executor_contract"] = dict(config["executor_contract"])
    drifted["executor_contract"]["probe"] = dict(probe)
    drifted["executor_contract"]["probe"]["pybullet_hz"] = 120
    import pytest
    with pytest.raises(ValueError, match="frozen probe"):
        agentic_ablation._validate_probe_protocol(drifted)


def test_agentic_policy_rows_freeze_requested_a0_to_a4_semantics() -> None:
    config = _load(POLICIES_PATH)
    assert [row["policy_id"] for row in config["policies"]] == [
        "A0", "A1", "A2", "A3", "A4"
    ]
    policies = _policy_map(config)

    assert policies["A0"]["initial_proposals_invoked"] == ["trellis"]
    assert policies["A0"]["selection"]["fixed_tool"] == "trellis"
    assert policies["A0"]["selection"]["may_read_evidence_for_choice"] is False

    assert config["fixed_priority"] == ["reconviagen", "trellis"]
    assert policies["A1"]["selection"]["strategy"] == "fixed_priority"
    assert policies["A1"]["selection"]["priority"] == config["fixed_priority"]
    assert policies["A1"]["selection"]["availability_definition"] == (
        "construction_candidate_completed_with_required_artifacts_and_evidence"
    )
    assert policies["A1"]["selection"]["fallback_to_trellis_only_when"] == [
        "reconviagen_typed_unavailable", "reconviagen_tool_crash"
    ]
    assert policies["A1"]["selection"]["may_read_evidence_for_choice"] is False
    assert policies["A1"]["selection"]["residual_or_gate_read_before_choice_forbidden"] is True

    for policy_id in ("A2", "A3", "A4"):
        assert policies[policy_id]["selection"]["strategy"] == "lexicographic"
        assert policies[policy_id]["selection"]["contract"] == config["lexicographic_selection"]["id"]
    assert config["lexicographic_selection"]["opaque_weighted_score_forbidden"] is True

    assert policies["A2"]["retry"]["enabled"] is False
    assert policies["A2"]["terminal_behavior"]["best_fails_primary_gate"]["support_label"] == "unsupported"
    assert policies["A3"]["retry"]["max_retries_per_job"] == 1
    assert policies["A3"]["terminal_behavior"]["best_fails_primary_gate_after_retry"] == {
        "decision": "accept",
        "support_label": "unsupported",
        "reason_code": "best_available_below_gate",
    }
    assert policies["A4"]["retry"]["max_retries_per_job"] == 1
    assert policies["A4"]["terminal_behavior"]["best_fails_primary_gate_after_retry"]["decision"] == "abstain"
    assert policies["A4"]["terminal_behavior"]["best_fails_primary_gate_after_retry"]["selected_asset"] is None

    retry = config["retry_contract"]
    assert retry["trigger"] == "all_available_initial_proposals_fail_primary_gate"
    assert retry["max_retries_per_job"] == 1
    assert retry["action_must_create_new_proposal_id"] is True
    assert retry["action_must_create_new_artifact_hash"] is True
    assert retry["require_new_transform_hash"] is True
    assert retry["preserve_trigger_failure_reason_codes"] is True
    assert retry["active_action_selection"] == "fixed_for_preliminary_v1"
    assert len(retry["rules"]) == 1
    registration = retry["rules"][0]
    assert registration["id"] == "all_initial_gate_failures"
    assert registration["when"] == "all_available_initial_proposals_fail_primary_gate"
    assert registration["action"] == "registration_signed_source_up_restart"
    assert registration["parameters"]["input_geometry"] == "lexicographically_best_initial_raw_mesh"
    assert registration["parameters"]["initial_hypothesis_already_used"] == "+z"
    assert registration["parameters"]["alternative_source_up_hypotheses"] == [
        "-z", "+x", "-x", "+y", "-y"
    ]
    assert registration["parameters"]["creates_new_transform_hash"] is True
    assert retry["inactive_future_actions"]["allowed_in_preliminary_v1"] is False
    assert config["controller_contract"]["initial_registration"]["source_up_hypotheses"] == ["+z"]

    ordering = config["lexicographic_selection"]["compare_in_order"]
    assert [row["field"] for row in ordering] == [
        "primary_gate_pass",
        "gate_failure_count",
        "symmetric_clipped_registration_residual_m",
        "scale_error_abs",
        "settle_drift_m",
        "tool_tiebreak_rank",
        "proposal_id",
    ]


def test_a4_registration_residual_sweep_is_fixed_and_one_dimensional() -> None:
    config = _load(POLICIES_PATH)
    sweep = config["a4_registration_residual_sweep"]
    assert sweep["policy_id"] == "A4"
    assert sweep["status"] == "development_preliminary"
    assert sweep["metric"] == "symmetric_clipped_registration_residual_m"
    assert sweep["values_m"] == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    assert sweep["primary_value_m"] == config["gates"]["max_registration_residual_m"]
    assert sweep["vary_only_this_check"] is True
    assert sweep["keep_all_other_gate_checks_fixed"] is True
    assert sweep["selection_after_sweep_forbidden"] is True


def test_agentic_jobs_static_population_and_provenance_contract() -> None:
    jobs = _load(JOBS_PATH)
    construction = _load(CONSTRUCTION_PATH)
    population = jobs["population"]
    scenes = population["scene_snapshots"]

    assert jobs["schema_version"] == 1
    assert jobs["study_scope"] == "conditional_fixed_gt_factory_jobs"
    assert jobs["paper_ready"] is False
    assert population["planned_scenes"] == 50
    assert population["planned_jobs_per_policy"] == 457
    assert population["planned_policy_object_rows"] == 2285
    assert population["policy_ids"] == ["A0", "A1", "A2", "A3", "A4"]
    assert len(scenes) == 50
    assert len({row["scene_id"] for row in scenes}) == 50
    assert [row["scene_id"] for row in scenes] == sorted(row["scene_id"] for row in scenes)
    expected_scene_ids = [str(value) for value in construction["population"]["scene_ids"]]
    assert [row["scene_id"] for row in scenes] == expected_scene_ids
    assert sum(row["accepted_jobs"] for row in scenes) == 457
    assert sum(row["reconviagen_available"] for row in scenes) == 453
    assert all(HEX64.fullmatch(row["report_sha256"]) for row in scenes)

    membership = population["membership"]
    assert membership["authority"] == "factory_report_json"
    assert membership["include_when"] == {"tier_in": ["A", "B"], "rejected_is_null": True}
    assert membership["controller_receives_report_fields"] is False
    assert any("hybrid_all.json" in value for value in membership["forbidden_membership_sources"])
    assert population["prehybrid_cross_check"] == {
        "path_template": "outputs/{scene_id}_factory/objects/aligned_all_prehybrid.json",
        "root_type": "list",
        "index_field": "index",
        "expected_rows_all_scenes": 789,
        "controller_access": "forbidden",
    }

    proposal_pool = jobs["proposal_pool"]
    assert proposal_pool["legacy_generation_provenance"] == {
        "artifact_bytes_hash_frozen": True,
        "original_generator_commits_recorded": False,
        "original_checkpoint_identities_recorded": False,
        "original_runtime_manifests_recorded": False,
        "generator_commit": None,
        "current_controller_commit_may_backfill_generator_commit": False,
        "claim_status": "preliminary_only",
    }
    assert proposal_pool["proposals"]["trellis"]["expected_available"] == 457
    rvg = proposal_pool["proposals"]["reconviagen"]
    assert rvg["expected_available"] == 453
    assert rvg["expected_typed_unavailable"] == 4
    assert len(rvg["typed_unavailable_job_ids"]) == 4
    for proposal in proposal_pool["proposals"].values():
        assert proposal["raw_mesh_path_template"].endswith("_mesh.ply")
        assert "aligned" not in proposal["raw_mesh_path_template"]
        assert "hybrid" not in proposal["raw_mesh_path_template"]
    assert proposal_pool["required_resolved_identity_fields"] == ["path", "size_bytes", "sha256"]
    canonical = proposal_pool["raw_mesh_manifest_canonicalization"]
    assert canonical == {
        "encoding": "utf-8",
        "serialization": "json",
        "document_shape": "array",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "terminal_newline": False,
        "one_manifest_per_tool": True,
        "path_base": "repository_root",
        "path_style": "relative_posix",
        "file_hash": "sha256_raw_file_bytes",
        "file_size": "stat_size_bytes",
        "tool_field_values": {"trellis": "trellis", "reconviagen": "reconviagen"},
        "row_fields": ["job_id", "tool", "path", "size_bytes", "sha256"],
        "row_order": ["job_id_utf8", "tool_utf8"],
    }
    assert all(
        HEX64.fullmatch(value["manifest_sha256"])
        for value in proposal_pool["raw_mesh_snapshots"].values()
    )

    boundary = jobs["controller_input_boundary"]
    assert boundary["reject_unknown_keys"] is True
    assert boundary["exact_sanitized_schemas"] == {
        "resolved_jobs": {
            "top_level_keys": [
                "schema_version", "freeze_id", "study_scope", "counts",
                "source_contract", "observation_protocol", "scenes",
            ],
            "counts_keys": ["scenes", "jobs", "policy_object_rows"],
            "source_contract_keys": [
                "jobs_sha256", "jobs_hash_method", "contract_sha256",
                "contract_hash_method", "contract_freeze_id", "code_commit",
            ],
            "scene_keys": [
                "scene_id", "source_scene_gaussian", "camera_artifacts", "jobs",
            ],
            "source_identity_keys": ["path", "size_bytes", "sha256"],
            "camera_artifacts_keys": ["intrinsics", "poses"],
            "camera_identity_keys": ["path", "size_bytes", "sha256"],
            "job_keys": [
                "freeze_id", "job_id", "scene_id", "object_slot",
                "artifact_paths", "artifact_hashes", "construction_evidence",
            ],
            "job_artifact_paths_keys": ["rgba"],
            "job_artifact_hashes_keys": ["rgba"],
            "job_construction_evidence_keys": [
                "source_frame", "bbox_px", "legacy_mask_provenance",
            ],
        },
        "resolved_proposals": {
            "top_level_keys": ["schema_version", "freeze_id", "counts", "proposals"],
            "counts_keys": [
                "jobs", "trellis_available", "reconviagen_available",
                "reconviagen_typed_unavailable",
            ],
            "proposal_keys": [
                "freeze_id", "job_id", "scene_id", "object_slot", "proposal_id",
                "tool_id", "availability", "artifact_paths", "artifact_hashes",
                "artifact_sizes", "construction_evidence", "raw_generator_provenance",
            ],
            "available_artifact_paths_keys": ["raw_mesh", "raw_gaussian"],
            "available_artifact_hashes_keys": ["raw_mesh", "raw_gaussian"],
            "available_artifact_sizes_keys": ["raw_mesh", "raw_gaussian"],
            "available_construction_evidence_keys": [],
            "typed_unavailable_artifact_paths_keys": [],
            "typed_unavailable_artifact_hashes_keys": [],
            "typed_unavailable_artifact_sizes_keys": [],
            "typed_unavailable_construction_evidence_keys": [
                "typed_failure", "missing_artifacts",
            ],
            "raw_generator_provenance_keys": [
                "artifact_bytes_hash_frozen", "generator_commit",
                "checkpoint_identity_recorded", "runtime_manifest_recorded",
                "claim_status",
            ],
        },
    }
    forbidden = boundary["forbidden_paths_or_patterns"]
    for name in ("gt_points.ply", "report.json", "aligned.json", "rvg_eval.json", "hybrid.json", "hybrid_all.json"):
        assert any(name in pattern for pattern in forbidden)
    assert boundary["fail_on_forbidden_path_or_field"] is True
    assert jobs["legacy_crop_provenance"] == {
        "total_jobs": 457,
        "sam3_masks_selected_by_overlap_with_gt_projection": 386,
        "gt_projected_mask_fallbacks": 71,
        "sum_must_equal_total": True,
        "claim_scope": "controller_gt_isolated_not_gt_free_discovery",
        "controller_may_read_mask_source": False,
    }
    smoke = jobs["smoke"]
    assert [row["id"] for row in smoke["object_scenarios"]] == [
        "both_initial_valid", "multiview_collapse", "retry_abstains"
    ]
    assert smoke["contract_scenarios"] == [{
        "id": "hard_export_impossible",
        "initial_proposal_pool": "empty",
        "expected_actions_include": ["reject", "abstain"],
    }]
    assert smoke["require_actions_observed"] == ["accept", "retry", "reject", "abstain"]


def test_agentic_live_report_roster_matches_frozen_snapshot_if_available() -> None:
    jobs = _load(JOBS_PATH)
    population = jobs["population"]
    first_report = REPO_ROOT / population["membership"]["report_path_template"].format(
        scene_id=population["scene_snapshots"][0]["scene_id"]
    )
    if not first_report.is_file():
        # The source pool is an external experiment artifact in a clean clone;
        # static contract tests above remain mandatory there.
        return

    rows: list[dict] = []
    prehybrid_rows = 0
    for snapshot in population["scene_snapshots"]:
        scene_id = snapshot["scene_id"]
        report_path = REPO_ROOT / population["membership"]["report_path_template"].format(
            scene_id=scene_id
        )
        report_bytes = report_path.read_bytes()
        assert hashlib.sha256(report_bytes).hexdigest() == snapshot["report_sha256"]
        report = json.loads(report_bytes)
        accepted = [
            row for row in report["objects"]
            if row.get("tier") in {"A", "B"} and row.get("rejected") is None
        ]
        assert len(accepted) == snapshot["accepted_jobs"]

        prehybrid_path = REPO_ROOT / population["prehybrid_cross_check"]["path_template"].format(
            scene_id=scene_id
        )
        prehybrid = json.loads(prehybrid_path.read_text(encoding="utf-8"))
        prehybrid_rows += len(prehybrid)
        prehybrid_indices = {int(row["index"]) for row in prehybrid}

        rvg_count = 0
        for row in accepted:
            index = int(row["index"])
            assert index in prehybrid_indices
            object_dir = REPO_ROOT / jobs["proposal_pool"]["object_dir_template"].format(
                scene_id=scene_id, object_index=index
            )
            trellis_complete = all(
                (object_dir / "trellis" / name).is_file()
                for name in ("trellis_mesh.ply", "trellis_gs.ply")
            )
            assert trellis_complete
            rvg_complete = all(
                (object_dir / "rvg" / name).is_file()
                for name in ("rvg_mesh.ply", "rvg_gs.ply")
            )
            rvg_count += int(rvg_complete)
            rows.append({
                "scene_id": scene_id,
                "object_index": index,
                "gt_object_id": row.get("gt_object_id"),
                "label": row["label"],
                "rvg_complete": rvg_complete,
            })
        assert rvg_count == snapshot["reconviagen_available"]

    assert prehybrid_rows == population["prehybrid_cross_check"]["expected_rows_all_scenes"]
    rows.sort(key=lambda row: (row["scene_id"], row["object_index"]))
    assert len(rows) == population["planned_jobs_per_policy"]
    assert sum(row["rvg_complete"] for row in rows) == 453
    assert hashlib.sha256(_canonical_json(rows)).hexdigest() == population["membership"]["resolved_roster_sha256"]


def test_agentic_observation_source_is_exact_and_gt_sanitized() -> None:
    jobs = _load(JOBS_PATH)
    source = jobs["observation_source"]
    assert source["expected_scenes"] == 50
    assert set(source["manifest_fields"]) == {
        "source_scene_gaussian", "camera_intrinsics", "camera_poses"
    }
    assert all(HEX64.fullmatch(value) for value in source["component_manifest_sha256"].values())
    assert HEX64.fullmatch(source["canonical_identity_manifest_sha256"])
    assert source["controller_forbidden_manifest_fields"] == [
        "gt", "source_images", "eval_frames", "methods"
    ]
    assert source["canonicalization"] == {
        "encoding": "utf-8",
        "serialization": "json",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "row_order": ["scene_id_utf8"],
    }
    assert source["controller_receives"] == {
        "representation": "nested_under_resolved_jobs_scenes",
        "scene_source_keys": ["scene_id", "source_scene_gaussian", "camera_artifacts"],
        "source_scene_gaussian_identity_keys": ["path", "size_bytes", "sha256"],
        "camera_artifacts_keys": ["intrinsics", "poses"],
        "camera_identity_keys": ["path", "size_bytes", "sha256"],
    }

    assert jobs["observation_protocol"] == {
        "render_mode": "RGB+ED",
        "rendered_rgb": "discarded",
        "rendered_channels_used": ["expected_depth", "gaussian_alpha"],
        "crop_channel_used": "rgba_alpha_only",
        "rgba_alpha_threshold_inclusive_uint8": 128,
        "gaussian_alpha_threshold_inclusive": 0.60,
        "depth_range_m_inclusive": [0.05, 8.0],
        "pixel_center_offset": 0.5,
        "camera_pose_input_convention": "colmap_world_to_camera",
        "camera_to_world_conversion": "matrix_inverse",
        "output_coordinate_frame": "world",
        "output_dtype": "float32",
        "output_unit": "metre",
        "forbidden_render_inputs": [
            "rendered_rgb", "evaluation_image", "evaluation_mask", "evaluation_geometry"
        ],
    }
    protocol = jobs["observation_protocol"]
    assert (
        protocol["gaussian_alpha_threshold_inclusive"]
        == agentic_ablation.OBSERVATION_ALPHA_MIN
    )
    assert protocol["depth_range_m_inclusive"] == [
        agentic_ablation.DEPTH_MIN_M,
        agentic_ablation.DEPTH_MAX_M,
    ]

    first_scene = jobs["population"]["scene_snapshots"][0]["scene_id"]
    first_manifest = REPO_ROOT / source["manifest_path_template"].format(scene_id=first_scene)
    if not first_manifest.is_file():
        return

    rows = []
    for snapshot in jobs["population"]["scene_snapshots"]:
        manifest_path = REPO_ROOT / source["manifest_path_template"].format(
            scene_id=snapshot["scene_id"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        camera = manifest["camera_artifacts"]
        rows.append({
            "scene_id": manifest["scene_id"],
            "source_scene_gaussian": {
                key: manifest["source_scene_gaussian"][key]
                for key in ("path", "size_bytes", "sha256")
            },
            "camera_intrinsics": {
                key: camera["intrinsics"][key]
                for key in ("path", "size_bytes", "sha256")
            },
            "camera_poses": {
                key: camera["poses"][key]
                for key in ("path", "size_bytes", "sha256")
            },
        })
    assert hashlib.sha256(_canonical_json(rows)).hexdigest() == source["canonical_identity_manifest_sha256"]
