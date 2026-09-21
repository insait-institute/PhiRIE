from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/icra2027/e4_collision_repair_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("e4_collision_repair_pilot_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_collision(module):
    slots = ["obj_00", "obj_01", "obj_02"]
    digest = "a" * 64
    return slots, {
        "background_carve": {
            "background_sha256": digest,
            "carved_slots": slots,
            "discovered_slots": slots,
            "geometry_source": "transformed_convex_collision_hulls",
            "hull_count": 9,
            "lower_support_margin_m": 0.0,
            "margin_m": 0.02,
            "mode": "paired_policy_union",
            "policy_ids": ["A0", "A4"],
            "roster_sha256": digest,
            "schema_version": 1,
            "source_mesh_sha256": digest,
            "specification_sha256": digest,
            "support_clip_source": "discovered_scan_aabb_bottom_plus_5mm",
        },
        "collision_exclusion": {
            "coacd_candidate_parts": 100,
            "coacd_parts_rejected_intrusion": 12,
            "emitted_coacd_intrusion_count": 0,
            "hull_count": 9,
            "primitive_intrusions": [],
            "schema_version": 1,
            "unresolved_intrusion_count": 0,
        },
    }


def test_pilot_roster_targets_large_informative_failed_scenes() -> None:
    module = _load_module()
    assert module.PILOT_SCENES == ("3db0a1c8f3", "d755b3d9d8")
    assert set(module.PILOT_SCENE_SELECTION) == set(module.PILOT_SCENES)
    assert module.POLICIES == ("A0", "A4")
    assert module.MODES == ("room", "shim")


def test_room_background_and_exclusion_contract_is_exact_and_passes() -> None:
    module = _load_module()
    slots, collision = _valid_collision(module)
    checks, summary = module._validate_background_contract(
        collision, discovered_slots=slots, accepted_slots=["obj_01"]
    )
    assert all(checks.values()), checks
    assert summary["hull_count"] == 9
    assert summary["coacd_parts_rejected_intrusion"] == 12
    assert summary["source_mesh_sha256"] == "a" * 64


def test_room_contract_fails_closed_on_intrusion_or_schema_extension() -> None:
    module = _load_module()
    slots, collision = _valid_collision(module)
    broken = copy.deepcopy(collision)
    broken["collision_exclusion"]["emitted_coacd_intrusion_count"] = 1
    broken["collision_exclusion"]["unresolved_intrusion_count"] = 1
    checks, _ = module._validate_background_contract(
        broken, discovered_slots=slots, accepted_slots=["obj_00"]
    )
    assert checks["no_emitted_coacd_intrusion"] is False
    assert checks["no_unresolved_intrusion"] is False

    extended = copy.deepcopy(collision)
    extended["background_carve"]["unsealed_extra"] = True
    checks, _ = module._validate_background_contract(
        extended, discovered_slots=slots, accepted_slots=["obj_00"]
    )
    assert checks["background_carve_exact_schema"] is False


def _comparisons(*, paired_room_slots=("obj_01", "obj_02"), shim_slots=()):
    result = {}
    for scene in ("3db0a1c8f3", "d755b3d9d8"):
        background = {
            "background_sha256": f"{scene:0<64}"[:64],
            "hull_count": 4,
            "specification_sha256": "b" * 64,
        }
        for policy in ("A0", "A4"):
            result[f"{scene}/{policy}"] = {
                "background": background,
                "comparison_passed": True,
                "room_stable_slots": list(paired_room_slots),
                "shim_stable_slots": list(shim_slots),
            }
    return result


def test_aggregate_releases_only_on_room_hard_gate() -> None:
    module = _load_module()
    gate = module.build_aggregate_gate(
        repair_id="repair-test",
        code={"commit": "a" * 40},
        comparisons=_comparisons(shim_slots=("obj_01", "obj_02", "obj_03")),
    )
    assert gate["repair_gate_passed"] is True
    assert gate["downstream_candidate_screen_allowed"] is True
    assert gate["gpu_launch_allowed"] is False
    assert gate["large_rollout_launch_allowed"] is False


def test_shim_cannot_unlock_insufficient_room_stability() -> None:
    module = _load_module()
    comparisons = _comparisons(
        paired_room_slots=("obj_01",),
        shim_slots=("obj_01", "obj_02", "obj_03", "obj_04"),
    )
    gate = module.build_aggregate_gate(
        repair_id="repair-test", code={"commit": "a" * 40}, comparisons=comparisons
    )
    assert gate["repair_gate_passed"] is False
    assert gate["downstream_candidate_screen_allowed"] is False
    assert gate["checks"][
        "each_pilot_scene_has_at_least_two_paired_stable_room_slots"
    ] is False


def test_inconsistent_a0_a4_background_blocks_release() -> None:
    module = _load_module()
    comparisons = _comparisons()
    comparisons["3db0a1c8f3/A4"]["background"] = {
        **comparisons["3db0a1c8f3/A4"]["background"],
        "background_sha256": "f" * 64,
    }
    gate = module.build_aggregate_gate(
        repair_id="repair-test", code={"commit": "a" * 40}, comparisons=comparisons
    )
    assert gate["repair_gate_passed"] is False
    assert gate["checks"]["common_background_identical_across_A0_A4_per_scene"] is False


def test_runner_passes_both_factories_only_to_room_export_and_seals_full_tree() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if mode == "room":' in source
    assert '"--background-carve-factory"' in source
    assert 'for paired_policy in POLICIES:' in source
    assert source.count('"sim", "sim_export"') >= 2
    assert "_tree_inventory(" in source
    assert "_revalidate_export_closure(" in source
    assert source.count("_revalidate_export_closure(") >= 3
    assert "validation summary differs from raw replay" in source
    assert "comparison summary differs from raw replay" in source


def test_mode_comparison_is_derived_from_validation_gates_not_copied_summary() -> None:
    module = _load_module()
    source = {
        "validation_passed": True,
        "source_binding": {"source": "same"},
        "accepted_slots": ["obj_01", "obj_02"],
        "discovered_slots": ["obj_00", "obj_01", "obj_02"],
    }
    room = {
        **source,
        "background": {"background_sha256": "a" * 64},
        "initial_contacts": {"max_static_penetration_m": 0.0},
        "stable_slots": ["obj_01"],
    }
    shim = {
        **source,
        "initial_contacts": {"max_static_penetration_m": 0.0},
        "stable_slots": ["obj_01", "obj_02"],
    }
    gate = module.build_mode_comparison_gate(
        repair_id="repair-test",
        scene="3db0a1c8f3",
        policy="A0",
        code={"commit": "a" * 40},
        validations={"room": room, "shim": shim},
        validation_bundle_sha256={"room": "b" * 64, "shim": "c" * 64},
        errors={},
    )
    assert gate["room_stable_slots"] == ["obj_01"]
    assert gate["room_stable_slot_count"] == 1
    assert gate["shim_stable_slot_count"] == 2
    assert gate["stable_count_delta_room_minus_shim"] == -1


def test_room_and_shim_floor_contracts_are_deliberately_distinct() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    room_block = source.split('if mode == "room":', 2)[2].split("else:", 1)[0]
    shim_block = source.split('if mode == "room":', 2)[2].split("else:", 1)[1]
    assert "scannetpp_world_frame_z0" in room_block
    assert "legacy_below_lowest_object" in shim_block
    assert "shim_preserves_legacy_floor_contract" in shim_block


def test_replay_comparator_tolerates_only_finite_subpicometer_float_drift() -> None:
    module = _load_module()
    baseline = {"nested": [1, {"metric": 0.125}], "exact": "binding"}
    within = copy.deepcopy(baseline)
    within["nested"][1]["metric"] += 1e-13
    beyond = copy.deepcopy(baseline)
    beyond["nested"][1]["metric"] += 2e-12
    nonfinite = copy.deepcopy(baseline)
    nonfinite["nested"][1]["metric"] = math.nan
    wrong_type = copy.deepcopy(baseline)
    wrong_type["nested"][1]["metric"] = "0.125"
    assert module._same_replayed_structure(baseline, within) is True
    assert module._same_replayed_structure(baseline, beyond) is False
    assert module._same_replayed_structure(baseline, nonfinite) is False
    assert module._same_replayed_structure(baseline, wrong_type) is False
