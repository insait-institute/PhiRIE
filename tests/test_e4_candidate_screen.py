import inspect
import json
import math

import pytest

from robo.eval import e4_candidate_screen as screen


def _footprint(cx, cy, hx=0.05, hy=0.05, z=0.75):
    return {
        "center_xy_m": [cx, cy],
        "half_extents_xy_m": [hx, hy],
        "lo_xy_m": [cx - hx, cy - hy],
        "hi_xy_m": [cx + hx, cy + hy],
        "position_world_m": [cx, cy, z],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def test_replay_comparison_is_structurally_exact_with_tiny_float_tolerance():
    original = {
        "engine": "mujoco",
        "footprints": {
            "obj_00": {
                "settled": _footprint(2.309825226786043, 0.5),
            }
        },
        "settle_drift_m": {"obj_00": 2.309825226786043},
        "step_count": 1000,
        "step_function": "mujoco.mj_step",
    }
    cross_host_roundoff = json.loads(json.dumps(original))
    cross_host_roundoff["footprints"]["obj_00"]["settled"]["center_xy_m"][0] = (
        2.3098252267860424
    )
    cross_host_roundoff["settle_drift_m"]["obj_00"] = 2.3098252267860424
    assert screen._same_replay_structure(original, cross_host_roundoff)

    meaningful_drift = json.loads(json.dumps(original))
    meaningful_drift["settle_drift_m"]["obj_00"] += 2e-12
    assert not screen._same_replay_structure(original, meaningful_drift)

    wrong_step_count = json.loads(json.dumps(original))
    wrong_step_count["step_count"] = 999
    assert not screen._same_replay_structure(original, wrong_step_count)

    wrong_numeric_type = json.loads(json.dumps(original))
    wrong_numeric_type["step_count"] = 1000.0
    assert not screen._same_replay_structure(original, wrong_numeric_type)

    nonfinite = json.loads(json.dumps(original))
    nonfinite["settle_drift_m"]["obj_00"] = math.nan
    assert not screen._same_replay_structure(original, nonfinite)

    wrong_schema = json.loads(json.dumps(original))
    wrong_schema["extra"] = None
    assert not screen._same_replay_structure(original, wrong_schema)


def _paired_footprints(rows):
    return {
        policy: {
            slot: {"nominal": rectangle, "settled": rectangle}
            for slot, rectangle in rows.items()
        }
        for policy in screen.POLICIES
    }


def _frozen_task(task_id, target):
    return {
        "any_instance": False,
        "instructions": {
            "default": "move the leftmost mouse to the designated region",
            "specific": "pick up the leftmost mouse and place it in the designated region",
            "vague": "move the leftmost mouse",
        },
        "receptacle": None,
        "region": {
            "cx": 0.75,
            "cy": 0.05,
            "hx": 0.10,
            "hy": 0.10,
            "zlo": 0.68,
            "zhi": 1.0,
        },
        "target": target,
        "target_label": "mouse",
        "task_id": task_id,
    }


def _expected_task_contracts(task_id, target):
    task = _frozen_task(task_id, target)
    slots = sorted([target, "obj_01"])
    return {
        task_id: {
            "any_instance": False,
            "eligible_same_label_by_policy": {
                policy: sorted([target, "obj_01"])
                for policy in screen.POLICIES
            },
            "qualifier": "leftmost",
            "runtime_body_slots_by_policy": {
                policy: slots
                for policy in screen.POLICIES
            },
            "runtime_dimensions_by_policy": {
                policy: {slot: [0.10, 0.10, 0.10] for slot in slots}
                for policy in screen.POLICIES
            },
            "runtime_nominal_positions_by_policy": {
                policy: {
                    target: [0.50, 0.05, 0.75],
                    "obj_01": [0.50, 0.0, 0.75],
                }
                for policy in screen.POLICIES
            },
            "task": task,
            "target": target,
            "target_label": "mouse",
            "workspace_suite_by_policy": {
                policy: {"robot": {"base_pos": [0.0, 0.0, 0.70], "base_yaw": 0.0}}
                for policy in screen.POLICIES
            },
        }
    }


def _metric_row(task_id, target, policy, episode):
    from robo.eval.episode_log import derive_reset_seed

    seed = derive_reset_seed(screen.BASE_SEED, task_id, episode)
    draw = screen.np.random.RandomState(seed).random_sample(2)
    identity = {
        "cell_id": (
            f"{policy.lower()}__{task_id}__seed{screen.BASE_SEED}__ep{episode}"
        ),
        "episode": episode,
        "policy_id": policy,
        "reset_seed": seed,
        "scene_id": screen.SCENE_IDS[0],
        "task_id": task_id,
        "target": target,
    }
    contract = _expected_task_contracts(task_id, target)[task_id]
    offset = (draw * 2.0 - 1.0) * screen.JITTER_XY_M
    current = {
        target: _footprint(0.50 + offset[0], 0.05 + offset[1]),
        "obj_01": _footprint(0.50, 0.0),
    }
    workspace = screen._workspace_from_footprints(
        current=current,
        task=contract["task"],
        suite=contract["workspace_suite_by_policy"][policy],
    )
    assert workspace["passed"] is True
    checks = {key: True for key in screen.CELL_CHECK_KEYS}
    return {
        **identity,
        "checks": checks,
        "passed": True,
        "qualifier": {
            **identity,
            "any_instance": False,
            "applicable": True,
            "competitor": "obj_01",
            "coordinates_in_robot_base_frame": {
                target: {
                    "forward_m": current[target]["center_xy_m"][0],
                    "left_m": current[target]["center_xy_m"][1],
                },
                "obj_01": {"forward_m": 0.50, "left_m": 0.0},
            },
            "eligible_same_label_objects": sorted([target, "obj_01"]),
            "observed_margin_m": current[target]["center_xy_m"][1],
            "passed": True,
            "qualifier": "leftmost",
            "qualifier_scores_m": {
                target: current[target]["center_xy_m"][1],
                "obj_01": 0.0,
            },
            "strict_required_margin_m": screen.QUALIFIER_MARGIN_M,
            "target_label": "mouse",
        },
        "reset_jitter": {
            "algorithm": "numpy.random.RandomState.random_sample_then_affine",
            "applied": True,
            "body": target,
            "max_abs_xy_m": screen.JITTER_XY_M,
            "offset_xy_m": ((draw * 2.0 - 1.0) * screen.JITTER_XY_M).tolist(),
            "reset_seed": seed,
            "uniform_draw_0_1": draw.tolist(),
        },
        "settle_contract_checks": {
            key: True for key in screen.SETTLE_CHECK_KEYS
        },
        "settle_protocol": {
            "engine": "mujoco",
            "model_timestep_s": 1.0 / 600.0,
            "requested_duration_s": screen.RESET_SETTLE_S,
            "simulated_duration_s": screen.RESET_SETTLE_S,
            "step_count": screen.EXPECTED_RESET_STEPS,
            "step_function": "mujoco.mj_step",
        },
        "stability": {
            "checks": {
                "all_body_linear_speed_below_50mm_s": True,
                "all_body_settle_drift_below_30mm": True,
                "target_settle_drift_below_30mm": True,
            },
            "drift_rows": [
                {
                    "drift_m": 0.0,
                    "expected_post_jitter_position_m": current[slot]["position_world_m"],
                    "observed_post_settle_position_m": current[slot]["position_world_m"],
                    "slot": slot,
                }
                for slot in sorted(current)
            ],
            "maximum_room_drift_m": 0.0,
            "maximum_room_linear_speed_m_s": 0.01,
            "passed": True,
            "speed_rows": [
                {
                    "linear_speed_m_s": 0.01,
                    "linear_velocity_xyz_m_s": [0.01, 0.0, 0.0],
                    "slot": slot,
                }
                for slot in sorted(current)
            ],
            "target_drift_m": 0.0,
        },
        "workspace": workspace,
    }


def test_workspace_replay_preserves_validated_serialized_rectangle_roundoff():
    task_id = "scene__obj_00_to_region"
    contract = _expected_task_contracts(task_id, "obj_00")[task_id]
    row = _metric_row(task_id, "obj_00", "A0", 0)
    footprints = json.loads(json.dumps(row["workspace"]["runtime_footprints"]))
    value = footprints["obj_00"]["half_extents_xy_m"][0]
    footprints["obj_00"]["half_extents_xy_m"][0] = screen.np.nextafter(value, math.inf)
    workspace = screen._workspace_from_footprints(
        current=footprints, task=contract["task"],
        suite=contract["workspace_suite_by_policy"]["A0"])
    screen._replay_workspace_record(workspace, contract=contract, policy="A0", label="roundoff")
    workspace["runtime_footprints"]["obj_00"]["half_extents_xy_m"][0] += 1e-8
    with pytest.raises(screen.CandidateScreenError, match="factory dimensions"):
        screen._replay_workspace_record(workspace, contract=contract, policy="A0", label="forgery")


def _metric_lattice(task_id="scene__obj_00_to_region", target="obj_00"):
    return [
        _metric_row(task_id, target, policy, episode)
        for policy in screen.POLICIES
        for episode in range(screen.EPISODES)
    ]


def test_candidate_roster_is_exact_and_exclusions_are_disjoint():
    assert screen.SCENE_IDS == (
        "3db0a1c8f3",
        "27dd4da69e",
        "d755b3d9d8",
        "acd95847c5",
        "40aec5fffa",
    )
    assert screen.REFERENCE_ONLY_SCENE_IDS == ("b0a08200c9",)
    assert screen.EXCLUDED_SCENE_IDS == (
        "825d228aec",
        "9071e139d9",
        "cc5237fd77",
    )
    assert not set(screen.SCENE_IDS) & set(screen.EXCLUDED_SCENE_IDS)
    assert not set(screen.SCENE_IDS) & set(screen.REFERENCE_ONLY_SCENE_IDS)


def test_reset_and_scientific_thresholds_are_frozen():
    assert screen.EXPORT_SETTLE_S == 2.0
    assert screen.EXPORT_SETTLE_STEPS == 1000
    assert screen.RESET_SETTLE_S == 1.5
    assert screen.EXPECTED_RESET_STEPS == 900
    assert screen.EXPORT_SETTLE_STEPS != screen.EXPECTED_RESET_STEPS
    assert inspect.signature(screen.snapshot_export_footprints).parameters[
        "steps"
    ].default == screen.EXPORT_SETTLE_STEPS
    assert screen.EPISODES == 5
    assert screen.JITTER_XY_M == 0.01
    assert screen.QUALIFIER_MARGIN_M == 0.03
    assert screen.MIN_OBSTACLE_CLEARANCE_M == 0.02
    assert screen.MIN_BASE_BODY_CLEARANCE_M == 0.02


def test_prepared_contract_accepts_candidate_order_different_from_frozen_order(
    tmp_path, monkeypatch
):
    first_id = "scene__obj_01_to_region"
    second_id = "scene__obj_00_to_region"
    candidate_order = [
        _frozen_task(second_id, "obj_00"),
        _frozen_task(first_id, "obj_01"),
    ]
    factory_rows = {}
    for slot in ("obj_00", "obj_01"):
        row = _eligible_object_row(drift=0.01)
        row["dims"] = screen.np.asarray(row["dims"], dtype=float)
        row["name"] = slot
        factory_rows[slot] = row
    nominal = {
        "obj_00": {"nominal": _footprint(0.50, 0.05)},
        "obj_01": {"nominal": _footprint(0.50, 0.00)},
    }
    prepared = {
        "factories": {
            policy: tmp_path / policy
            for policy in screen.POLICIES
        },
        "gate": {
            "candidate_count": 2,
            "footprint_replays": {
                policy: {"footprints": nominal}
                for policy in screen.POLICIES
            },
        },
        "suites": {
            policy: {
                "exclude_objects": [],
                "robot": {"base_pos": [0.0, 0.0, 0.70], "base_yaw": 0.0},
                "tasks": list(candidate_order),
            }
            for policy in screen.POLICIES
        },
        "task_bundle": {"logical_task_ids": [first_id, second_id]},
    }
    monkeypatch.setattr(screen, "_rows_for_factory", lambda _factory: factory_rows)

    contracts = screen._prepared_task_contracts(prepared)

    assert list(contracts) == [first_id, second_id]
    assert contracts[first_id]["target"] == "obj_01"
    assert contracts[second_id]["target"] == "obj_00"


def test_reach_envelope_requires_radial_and_cone_safety_margins():
    safe = screen.reach_envelope([0.50, 0.0], [0.0, 0.0], 0.0)
    assert safe["passed"] is True
    assert safe["safe"] is True
    radial_edge = screen.reach_envelope([0.79, 0.0], [0.0, 0.0], 0.0)
    assert radial_edge["passed"] is True
    assert radial_edge["safe"] is False
    cone_edge = screen.reach_envelope(
        [0.5 * math.cos(math.radians(58)), 0.5 * math.sin(math.radians(58))],
        [0.0, 0.0],
        0.0,
    )
    assert cone_edge["passed"] is True
    assert cone_edge["safe"] is False


def test_shared_base_checks_both_arms_nominal_and_settled_footprints():
    footprints = _paired_footprints({"obj_00": _footprint(0.60, 0.0)})
    footprints["A4"]["obj_00"]["settled"] = _footprint(0.21, 0.0, 0.01, 0.01)
    result = screen.base_clearance([0.0, 0.0], footprints)
    assert result["passed"] is False
    assert any(
        row["policy_id"] == "A4"
        and row["slot"] == "obj_00"
        and row["state"] == "settled"
        for row in result["blocking_footprints"]
    )


def test_shared_base_clearance_is_edge_gap_after_robot_radius():
    footprints = _paired_footprints({"obj_00": _footprint(0.26, 0.0, 0.05, 0.05)})
    result = screen.base_clearance([0.0, 0.0], footprints)
    assert result["minimum_clearance_m"] == pytest.approx(0.03)
    assert result["passed"] is True


def test_region_rejects_non_target_obstacle_from_either_arm():
    footprints = _paired_footprints(
        {
            "obj_00": _footprint(-0.4, 0.0),
            "obj_01": _footprint(0.17, 0.0, 0.04, 0.04),
        }
    )
    footprints["A0"]["obj_01"]["settled"] = _footprint(0.14, 0.0, 0.04, 0.04)
    result = screen.region_obstacle_clearance(
        {"cx": 0.0, "cy": 0.0, "hx": 0.10, "hy": 0.10},
        target_slot="obj_00",
        footprints=footprints,
    )
    assert result["passed"] is False
    assert any(
        row["policy_id"] == "A0"
        and row["slot"] == "obj_01"
        and row["state"] == "settled"
        for row in result["blocking_footprints"]
    )


def test_region_ignores_only_its_target_and_records_minimum_clearance():
    footprints = _paired_footprints(
        {
            "obj_00": _footprint(0.0, 0.0),
            "obj_01": _footprint(0.30, 0.0, 0.05, 0.05),
        }
    )
    result = screen.region_obstacle_clearance(
        {"cx": 0.0, "cy": 0.0, "hx": 0.10, "hy": 0.10},
        target_slot="obj_00",
        footprints=footprints,
    )
    assert result["minimum_clearance_m"] == pytest.approx(0.15)
    assert result["passed"] is True


def test_support_table_unions_different_aabbs_for_same_slot_in_both_arms():
    rows_by_policy = {
        "A0": {
            "obj_00": {
                "aabb": [[-1.0, -2.0, 0.90], [0.0, 1.0, 1.10]],
                "bottom_z": 0.97,
                "name": "obj_00",
            }
        },
        "A4": {
            "obj_00": {
                "aabb": [[3.0, -1.0, 0.90], [5.0, 4.0, 1.10]],
                "bottom_z": 1.00,
                "name": "obj_00",
            }
        },
    }
    table = screen._table_for_cluster(["obj_00"], rows_by_policy)

    assert table["cx"] - table["hx"] == pytest.approx(-1.30)
    assert table["cx"] + table["hx"] == pytest.approx(5.30)
    assert table["cy"] - table["hy"] == pytest.approx(-2.30)
    assert table["cy"] + table["hy"] == pytest.approx(4.30)
    assert table["top_z"] == pytest.approx(0.95)
    assert table["member_slots"] == ["obj_00"]
    assert table["policy_slot_members"] == [
        {"policy_id": "A0", "slot": "obj_00"},
        {"policy_id": "A4", "slot": "obj_00"},
    ]


def test_large_anisotropic_target_has_positive_axiswise_eroded_region_core():
    footprints = _paired_footprints(
        {"obj_00": _footprint(0.50, 0.0, hx=0.14, hy=0.03)}
    )
    table = {"cx": 0.50, "cy": 0.0, "hx": 1.0, "hy": 1.0, "top_z": 0.70}

    options = screen._region_candidates(
        "obj_00", table=table, footprints=footprints
    )
    selected, _rejections = screen._region_for_base(
        options, base_xy=[0.20, 0.0], base_yaw=0.0
    )

    assert options
    assert options[0]["region"]["hx"] == pytest.approx(0.30)
    assert options[0]["region"]["hy"] == pytest.approx(0.30)
    assert selected is not None
    assert selected["eroded_core_half_extents_m"][0] > 0.0
    assert selected["eroded_core_half_extents_m"][1] > 0.0
    assert selected["eroded_core_safe_count"] >= 5


def _eligible_object_row(*, drift):
    return {
        "aabb": [[0.45, -0.05, 0.70], [0.55, 0.05, 0.80]],
        "bottom_z": 0.70,
        "center": [0.50, 0.0, 0.75],
        "dims": [0.10, 0.10, 0.10],
        "drift": drift,
        "label": "mouse",
        "mass": 0.10,
        "name": "obj_00",
        "tier": "A",
    }


def test_export_target_drift_requires_strictly_below_30mm():
    assert screen._eligibility_reasons(_eligible_object_row(drift=0.029999)) == []
    assert screen._eligibility_reasons(_eligible_object_row(drift=0.03)) == [
        "export_target_drift_not_below_30mm"
    ]


def test_export_drift_binding_rejects_row_that_disagrees_with_1000_step_replay():
    rows = {
        policy: {"obj_00": _eligible_object_row(drift=0.01)}
        for policy in screen.POLICIES
    }
    replays = {}
    for policy in screen.POLICIES:
        settled_x = 0.51 if policy == "A0" else 0.60
        footprints = {
            "obj_00": {
                "nominal": _footprint(0.50, 0.0),
                "settled": _footprint(settled_x, 0.0),
            }
        }
        replays[policy] = {
            "footprints": footprints,
            "settle_drift_m": {"obj_00": abs(settled_x - 0.50)},
        }

    with pytest.raises(
        screen.CandidateScreenError,
        match="exporter drift differs from 1000-step replay",
    ):
        screen._validate_export_drift_binding(
            rows_by_policy=rows,
            footprint_replays=replays,
        )


@pytest.mark.parametrize("rejecting_policy", screen.POLICIES)
def test_export_target_drift_failure_in_either_arm_rejects_paired_candidate(
    rejecting_policy,
):
    rows_by_policy = {
        policy: {
            "obj_00": _eligible_object_row(
                drift=0.03 if policy == rejecting_policy else 0.029999
            )
        }
        for policy in screen.POLICIES
    }
    plan = screen.plan_paired_region_tasks(
        scene_id=screen.SCENE_IDS[0],
        rows_by_policy=rows_by_policy,
        footprints=_paired_footprints({"obj_00": _footprint(0.50, 0.0)}),
    )

    assert plan["candidate_count"] == 0
    assert plan["all_object_candidates"]["obj_00"]["paired_eligible"] is False
    assert plan["all_object_candidates"]["obj_00"][
        "eligibility_reasons_by_policy"
    ][rejecting_policy] == ["export_target_drift_not_below_30mm"]


def test_qualifier_metric_replay_recomputes_exact_lattice_and_summary_extrema():
    task_id = "scene__obj_00_to_region"
    rows = _metric_lattice(task_id)

    replay = screen._replay_qualifier_metrics(
        rows,
        scene_id=screen.SCENE_IDS[0],
        expected_task_contracts=_expected_task_contracts(task_id, "obj_00"),
    )

    assert replay["cell_count"] == 10
    assert replay["exact_900_step_cells"] == 10
    assert replay["strict_pass_task_ids"] == [task_id]
    assert replay["task_summaries"] == [{
        "cell_count": 10,
        "cells_failed": 0,
        "cells_passed": 10,
        "maximum_room_drift_m": 0.0,
        "minimum_base_clearance_m": min(
            row["workspace"]["base_clearance"]["minimum_clearance_m"]
            for row in rows
        ),
        "minimum_goal_obstacle_clearance_m": min(
            row["workspace"]["obstacle_clearance"]["minimum_clearance_m"]
            for row in rows
        ),
        "minimum_qualifier_margin_m": min(
            row["qualifier"]["observed_margin_m"] for row in rows
        ),
        "strict_pass": True,
        "task_id": task_id,
    }]


@pytest.mark.parametrize(
    "forgery",
    (
        "stability",
        "workspace_base",
        "workspace_obstacle",
        "qualifier",
        "qualifier_unique",
        "settle",
    ),
)
def test_qualifier_metric_replay_rejects_raw_values_forged_behind_true_checks(
    forgery,
):
    task_id = "scene__obj_00_to_region"
    rows = _metric_lattice(task_id)
    row = rows[0]
    if forgery == "stability":
        row["stability"]["drift_rows"][0]["drift_m"] = 0.50
        row["stability"]["maximum_room_drift_m"] = 0.50
        row["stability"]["target_drift_m"] = 0.50
    elif forgery == "workspace_base":
        row["workspace"]["base_clearance"]["minimum_clearance_m"] = -1.0
    elif forgery == "workspace_obstacle":
        row["workspace"]["obstacle_clearance"]["minimum_clearance_m"] = -1.0
    elif forgery == "qualifier":
        row["qualifier"]["observed_margin_m"] = None
    elif forgery == "qualifier_unique":
        row["qualifier"].update({
            "applicable": False,
            "eligible_same_label_objects": ["obj_00"],
            "observed_margin_m": None,
            "qualifier": None,
        })
    else:
        row["settle_protocol"]["step_count"] = 899

    with pytest.raises(screen.CandidateScreenError):
        screen._replay_qualifier_metrics(
            rows,
            scene_id=screen.SCENE_IDS[0],
            expected_task_contracts=_expected_task_contracts(task_id, "obj_00"),
        )


def test_unique_qualifier_record_rejects_residual_competitor_fields():
    task_id = "scene__obj_00_to_region"
    contracts = _expected_task_contracts(task_id, "obj_00")
    contracts[task_id]["qualifier"] = None
    for policy in screen.POLICIES:
        contracts[task_id]["eligible_same_label_by_policy"][policy] = ["obj_00"]
    rows = _metric_lattice(task_id)
    for row in rows:
        row["qualifier"].update({
            "applicable": False,
            "eligible_same_label_objects": ["obj_00"],
            "observed_margin_m": None,
            "qualifier": None,
        })

    with pytest.raises(
        screen.CandidateScreenError, match="conditional schema differs"
    ):
        screen._replay_qualifier_metrics(
            rows,
            scene_id=screen.SCENE_IDS[0],
            expected_task_contracts=contracts,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "raw_stability",
        "forged_unique",
        "qualifier_workspace_cross_binding",
        "footprint_factory_dimensions",
    ),
)
def test_qualifier_validator_rejects_forged_strict_after_reseal(
    tmp_path, monkeypatch, tamper
):
    screen_id = f"e4-test-{tamper}"
    scene_id = screen.SCENE_IDS[0]
    task_id = "scene__obj_00_to_region"
    target = "obj_00"
    expected_commit = "a" * 40
    prepare_sha256 = "b" * 64
    root = tmp_path.resolve()
    expected_task_contracts = _expected_task_contracts(task_id, target)
    rows = _metric_lattice(task_id, target)
    replay = screen._replay_qualifier_metrics(
        rows,
        scene_id=scene_id,
        expected_task_contracts=expected_task_contracts,
    )
    code = {
        "code_root": str(screen.CODE_ROOT),
        "commit": expected_commit,
        "dirty": False,
    }
    menagerie = {
        "access_mode": "read_only_input; gate performs no writes",
        "closure_sha256": "c" * 64,
        "file_count": 94,
        "root": str(screen.EXPECTED_MENAGERIE_ROOT),
        "size_bytes": 40_901_071,
        "source_commit": screen.EXPECTED_MENAGERIE_COMMIT,
        "source_root": "/frozen/source",
    }
    gate = {
        "candidate_count": 1,
        "cell_count": replay["cell_count"],
        "code": code,
        "created_utc": "2026-09-04T00:00:00Z",
        "evidence_root": str(root),
        "exact_900_step_cells": replay["exact_900_step_cells"],
        "gpu_launch_allowed": False,
        "headline_eligible": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_candidate_scene_qualifier_gate",
        "menagerie": menagerie,
        "paper_ready": False,
        "prepare_manifest_sha256": prepare_sha256,
        "renderer_constructed": False,
        "scene_id": scene_id,
        "screen_id": screen_id,
        "semantic_blockers": replay["semantic_blockers"],
        "strict_pass_task_count": replay["strict_pass_task_count"],
        "strict_pass_task_ids": replay["strict_pass_task_ids"],
        "study_scope": screen.STUDY_SCOPE,
        "task_summaries": replay["task_summaries"],
    }
    frozen_task = _frozen_task(task_id, target)
    nominal_footprints = {
        target: {"nominal": _footprint(0.50, 0.05)},
        "obj_01": {"nominal": _footprint(0.50, 0.0)},
    }
    prepared = {
        "factories": {policy: root / policy for policy in screen.POLICIES},
        "gate": {
            "candidate_count": 1,
            "footprint_replays": {
                policy: {"footprints": nominal_footprints}
                for policy in screen.POLICIES
            },
        },
        "prepare_bundle": {"manifest_sha256": prepare_sha256},
        "suites": {
            policy: {
                "exclude_objects": [],
                "robot": {"base_pos": [0.0, 0.0, 0.70], "base_yaw": 0.0},
                "tasks": [frozen_task],
            }
            for policy in screen.POLICIES
        },
        "task_bundle": {"logical_task_ids": [task_id]},
    }
    factory_rows = {}
    for slot in (target, "obj_01"):
        row = _eligible_object_row(drift=0.01)
        row["dims"] = screen.np.asarray(row["dims"], dtype=float)
        row["name"] = slot
        factory_rows[slot] = row
    monkeypatch.setattr(screen, "_load_prepare", lambda **_kwargs: prepared)
    monkeypatch.setattr(screen, "_rows_for_factory", lambda _factory: factory_rows)
    monkeypatch.setattr(screen, "_frozen_menagerie_summary", lambda: menagerie)
    output_dir = (
        screen._experiment_root(root, screen_id)
        / "scene_qualifiers"
        / scene_id
    )
    screen._publish_bundle(
        output_dir,
        manifest_kind="e4_candidate_scene_qualifier_artifacts",
        payloads={
            "gate.json": screen._json_bytes(gate),
            "metrics.jsonl": screen._jsonl_bytes(rows),
        },
        manifest_fields={
            "code": code,
            "created_utc": "2026-09-04T00:00:00Z",
            "prepare_manifest_sha256": prepare_sha256,
            "scene_id": scene_id,
            "screen_id": screen_id,
        },
    )
    screen._validate_qualifier_output(
        root=root,
        screen_id=screen_id,
        scene_id=scene_id,
        expected_commit=expected_commit,
    )

    if tamper == "raw_stability":
        rows[0]["stability"]["drift_rows"][0]["drift_m"] = 0.50
        rows[0]["stability"]["maximum_room_drift_m"] = 0.50
        rows[0]["stability"]["target_drift_m"] = 0.50
    elif tamper == "forged_unique":
        for row in rows:
            qualifier = row["qualifier"]
            qualifier["eligible_same_label_objects"] = [target]
            qualifier["qualifier"] = None
            qualifier["applicable"] = False
            qualifier["observed_margin_m"] = None
            for key in (
                "competitor",
                "coordinates_in_robot_base_frame",
                "qualifier_scores_m",
            ):
                qualifier.pop(key)
    elif tamper == "qualifier_workspace_cross_binding":
        row = rows[0]
        current = dict(row["workspace"]["runtime_footprints"])
        target_pose = list(current[target]["position_world_m"])
        target_pose[1] = 0.02
        current[target] = screen._world_xy_rectangle(
            screen.np.asarray(target_pose, dtype=float),
            screen.np.asarray(current[target]["quaternion_wxyz"], dtype=float),
            screen.np.asarray([0.10, 0.10, 0.10], dtype=float),
        )
        row["workspace"] = screen._workspace_from_footprints(
            current=current,
            task=expected_task_contracts[task_id]["task"],
            suite=expected_task_contracts[task_id]["workspace_suite_by_policy"]["A0"],
        )
        assert row["workspace"]["passed"] is True
    else:
        row = rows[0]
        current = dict(row["workspace"]["runtime_footprints"])
        current["obj_01"] = _footprint(0.872, 0.05, hx=0.001, hy=0.001)
        row["workspace"] = screen._workspace_from_footprints(
            current=current,
            task=expected_task_contracts[task_id]["task"],
            suite=expected_task_contracts[task_id]["workspace_suite_by_policy"]["A0"],
        )
        assert row["workspace"]["passed"] is True
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_bytes(screen._jsonl_bytes(rows))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metrics.jsonl"] = screen._identity_without_path(metrics_path)
    manifest_path.write_bytes(screen._json_bytes(manifest))
    seal_path = output_dir / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["members"]["manifest.json"] = screen._identity_without_path(manifest_path)
    seal_path.write_bytes(screen._json_bytes(seal))

    with pytest.raises(screen.CandidateScreenError):
        screen._validate_qualifier_output(
            root=root,
            screen_id=screen_id,
            scene_id=scene_id,
            expected_commit=expected_commit,
        )


def test_candidate_module_only_resolves_existing_scorer_region_without_scoring():
    import ast
    source = inspect.getsource(screen)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Name) and node.func.id == "TaskScorer"
                   for node in calls)
    scorer_calls = [node.func.attr for node in calls
                    if isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "TaskScorer"]
    assert scorer_calls == ["_resolve_region"]
    assert "gpu_launch_allowed" in source or "GPU" in source


def test_cli_exposes_explicit_phases_and_winner_smoke():
    parser = screen._parser()
    action = next(a for a in parser._actions if a.dest == "command")
    assert set(action.choices) == {
        "materialize",
        "prepare-winner-smoke",
        "prepare-automatic-candidates", "prepare-automatic-tasks",
        "prepare-scene",
        "qualify-scene",
        "aggregate",
    }
