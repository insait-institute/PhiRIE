from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run/icra2027/e4_robust_floor_support_diagnostic.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("e4_robust_floor_support", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _component(component_id: int, selected_as: str) -> dict:
    is_floor = selected_as == "floor"
    return {
        "component_id": component_id,
        "face_count": 20,
        "area_m2": 2.0 if is_floor else 0.06,
        "z50_m": 0.36 if is_floor else 0.80,
        "weighted_p95_residual_m": 0.01,
        "max_vertex_residual_m": 0.03 if is_floor else 0.01,
        "span_xy_m": [2.0, 2.0] if is_floor else [0.2, 0.2],
        "bounds_xy_m": (
            [[0.0, 0.0], [2.0, 2.0]]
            if is_floor
            else [[0.0, 0.0], [0.2, 0.2]]
        ),
        "projected_aabb_fill": 0.8,
        "projected_overlap_method": (
            "exact_triangle_union_clipped_to_object_aabb"
        ),
        "box_footprint_safe": True,
        "box_false_overlap_rejections": [],
        "eligible_floor": is_floor,
        "support_associations": (
            []
            if is_floor
            else [
                {
                    "slot": "obj_00",
                    "overlap_xy_m2": 0.01,
                    "overlap_object_fraction": 0.5,
                    "scan_bottom_minus_z50_m": 0.01,
                }
            ]
        ),
        "selected_as": selected_as,
    }


def _collision(module, policy: str) -> dict:
    support_mode = (
        "legacy_max_residual"
        if policy == "robust_floor_only"
        else "robust_associated_weighted_p95"
    )
    return {
        "n_floor": 1,
        "n_supports": 1,
        "support_boxes": 1,
        "floor_z_m": 0.36,
        "floor_source": "robust_area_weighted_component",
        "collision_exclusion": {
            "unresolved_intrusion_count": 0,
            "primitive_intrusions": [],
            "emitted_coacd_intrusion_count": 0,
        },
        "floor_patch_selection": {
            "expected_z_m": None,
            "tolerance_m": 0.02,
            "detected_z_m": 0.36,
            "status": "robust_matched",
            "selection_policy": policy,
            "selected_component_id": 1,
        },
        "raw_diagnostics": {
            "floor_aware_common_hulls": [{"name": "A0/obj_00/part_00.obj"}],
            "room_collision": {
                "support_rejections": [],
                "robust_support_demotion": {
                    "rejected_support_count": 0,
                    "demoted_source_face_count": 0,
                    "residual_rebuild_face_count": 7,
                    "face_inventory_validated": True,
                },
                "robust_emission": {
                    "floor_representation": "global_plane_only",
                    "primitive_intrusion_tolerance_m": 0.005,
                    "room_surface_policy": policy,
                },
                "primitive_intrusions_before_policy": [],
            },
            "dynamic_object_hulls": [
                {
                    "name": "obj_00/part_00.obj",
                    "floor_aware_clip": {"status": "selected"},
                }
            ],
            "feature_extraction": {
                "residual_triangle_count": 7,
                "robust_surfaces": {
                    "schema_version": 1,
                    "policy": policy,
                    "guards": deepcopy(module.EXPECTED_ROBUST_GUARDS),
                    "floor": {
                        "candidate_count": 2,
                        "eligible_count": 1,
                        "selected_component_id": 1,
                        "area_m2": 2.0,
                        "z50_m": 0.36,
                        "weighted_p95_residual_m": 0.01,
                        "span_xy_m": [2.0, 2.0],
                        "bounds_xy_m": [[0.0, 0.0], [2.0, 2.0]],
                        "runner_up_eligible_area_m2": None,
                        "area_ratio_to_next_eligible": None,
                        "dominance_passed": True,
                    },
                    "supports": {
                        "selection_mode": support_mode,
                        "candidate_count": 1,
                        "selected_count": 1,
                        "selected_component_ids": [2],
                        "rejected_count": 0,
                        "emitted_count": 1,
                        "rejected_component_ids": [],
                        "emitted_component_ids": [2],
                        "associated_slots": ["obj_00"],
                        "emitted_associated_slots": ["obj_00"],
                    },
                    "components": [
                        _component(1, "floor"),
                        _component(2, "support"),
                    ],
                }
            },
        },
    }


def _reject_support(report: dict) -> None:
    supports = report["raw_diagnostics"]["feature_extraction"][
        "robust_surfaces"
    ]["supports"]
    supports.update(
        rejected_count=1,
        emitted_count=0,
        rejected_component_ids=[2],
        emitted_component_ids=[],
        emitted_associated_slots=[],
    )
    components = report["raw_diagnostics"]["feature_extraction"][
        "robust_surfaces"
    ]["components"]
    components[1]["selected_as"] = "residual"
    report["n_supports"] = 0
    report["support_boxes"] = 0
    room = report["raw_diagnostics"]["room_collision"]
    room["support_rejections"] = [
        {
            "geom": "room_support_0",
            "reason": "penetration_depth_gt_0.005_m",
            "protected_hull_ids": ["A0/obj_00/part_00.obj"],
            "penetration_depths_m": [
                {
                    "protected_hull_id": "A0/obj_00/part_00.obj",
                    "depth_m": 0.006,
                }
            ],
            "max_penetration_depth_m": 0.006,
            "source_face_count": 20,
            "source_area_m2": 0.06,
        }
    ]
    room["primitive_intrusions_before_policy"] = [
        {
            "geom": "room_support_0",
            "exclusions": ["A0/obj_00/part_00.obj"],
        }
    ]
    room["robust_support_demotion"] = {
        "rejected_support_count": 1,
        "demoted_source_face_count": 20,
        "residual_rebuild_face_count": 27,
        "face_inventory_validated": True,
    }


def _surface_spec(policy: str) -> dict:
    return {
        "room_surface_policy": policy,
        "min_support_area_m2": 0.05,
        "plane_residual_tol_m": 0.02,
    }


def _append_box_false_overlap_component(report: dict) -> dict:
    robust = report["raw_diagnostics"]["feature_extraction"]["robust_surfaces"]
    component = _component(3, "residual")
    component.update(
        support_associations=[],
        box_footprint_safe=False,
        box_false_overlap_rejections=[
            {
                "slot": "obj_00",
                "emitted_box_overlap_xy_m2": 0.01,
                "emitted_box_overlap_object_fraction": 0.5,
                "projected_overlap_xy_m2": 0.001,
                "projected_overlap_object_fraction": 0.05,
                "scan_bottom_minus_z50_m": 0.01,
            }
        ],
    )
    robust["components"].append(component)
    robust["floor"]["candidate_count"] = 3
    robust["supports"]["candidate_count"] = 2
    return component


def test_frozen_v2_matrix_is_exact() -> None:
    module = _load_runner()
    assert module.VARIANT_IDS == ("3db-f", "3db-fs", "d755-f", "d755-fs")
    assert {row["schema_version"] for row in module.VARIANTS.values()} == {2}
    assert {
        row["room_surface_policy"] for row in module.VARIANTS.values()
    } == {"robust_floor_only", "robust_floor_plus_support"}
    assert all(row["hull_bottom"] == "clip_scan_aabb_bottom" for row in module.VARIANTS.values())
    assert all(row["intrusive_primitive"] == "fail" for row in module.VARIANTS.values())
    assert all(row["plane_residual_tol_m"] == 0.02 for row in module.VARIANTS.values())
    assert all(row["support_z"] == "fitted_mean_20mm" for row in module.VARIANTS.values())
    assert module.EXPECTED_ROBUST_GUARDS[
        "support_box_min_half_extent_xy_m"
    ] == 0.01
    assert module.EXPECTED_ROBUST_GUARDS[
        "support_projected_overlap_method"
    ] == "exact_triangle_union_clipped_to_object_aabb"
    assert module.EXPECTED_ROBUST_GUARDS[
        "support_box_false_overlap_policy"
    ] == "reject_component"


@pytest.mark.parametrize(
    "policy", ("robust_floor_only", "robust_floor_plus_support")
)
def test_robust_surface_contract_accepts_exact_frozen_report(policy: str) -> None:
    module = _load_runner()
    spec = _surface_spec(policy)
    assert module._robust_surface_contract(
        _collision(module, policy), spec, ["obj_00"]
    )


def test_robust_surface_contract_accepts_lossless_fs_support_rejection() -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    _reject_support(report)
    assert module._robust_surface_contract(
        report,
        _surface_spec("robust_floor_plus_support"),
        ["obj_00"],
    )


@pytest.mark.parametrize(
    "policy", ("robust_floor_only", "robust_floor_plus_support")
)
def test_robust_surface_contract_accepts_exact_box_false_overlap_rejection(
    policy: str,
) -> None:
    module = _load_runner()
    report = _collision(module, policy)
    _append_box_false_overlap_component(report)
    assert module._robust_surface_contract(
        report, _surface_spec(policy), ["obj_00"]
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda row: row.update(box_footprint_safe=True),
        lambda row: row["box_false_overlap_rejections"][0].update(slot="obj_99"),
        lambda row: row["box_false_overlap_rejections"][0].update(
            emitted_box_overlap_xy_m2=0.001
        ),
        lambda row: row["box_false_overlap_rejections"][0].update(
            emitted_box_overlap_object_fraction=0.09
        ),
        lambda row: row["box_false_overlap_rejections"][0].update(
            projected_overlap_xy_m2=0.002,
            projected_overlap_object_fraction=0.1,
        ),
        lambda row: row["box_false_overlap_rejections"][0].update(
            scan_bottom_minus_z50_m=0.031
        ),
        lambda row: row["box_false_overlap_rejections"][0].update(extra=True),
    ),
)
def test_robust_surface_contract_rejects_invalid_box_false_overlap_raw_schema(
    mutator,
) -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    row = _append_box_false_overlap_component(report)
    mutator(row)
    assert not module._robust_surface_contract(
        report, _surface_spec("robust_floor_plus_support"), ["obj_00"]
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda row: row.update(projected_overlap_method="aabb_overlap"),
        lambda row: row["support_associations"][0].update(overlap_xy_m2=0.001),
        lambda row: row["support_associations"][0].update(
            overlap_object_fraction=0.09
        ),
        lambda row: row["support_associations"][0].update(
            scan_bottom_minus_z50_m=-0.006
        ),
        lambda row: row["support_associations"][0].update(slot="obj_99"),
    ),
)
def test_robust_surface_contract_rejects_invalid_exact_overlap_association(
    mutator,
) -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    row = report["raw_diagnostics"]["feature_extraction"]["robust_surfaces"][
        "components"
    ][1]
    mutator(row)
    assert not module._robust_surface_contract(
        report, _surface_spec("robust_floor_plus_support"), ["obj_00"]
    )


def test_robust_surface_contract_recomputes_every_floor_eligibility() -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    robust = report["raw_diagnostics"]["feature_extraction"]["robust_surfaces"]
    support = robust["components"][1]
    support["eligible_floor"] = True
    robust["floor"].update(
        eligible_count=2,
        runner_up_eligible_area_m2=support["area_m2"],
        area_ratio_to_next_eligible=(
            robust["floor"]["area_m2"] / support["area_m2"]
        ),
    )
    assert not module._robust_surface_contract(
        report, _surface_spec("robust_floor_plus_support"), ["obj_00"]
    )


@pytest.mark.parametrize(
    ("policy", "metric", "value"),
    (
        ("robust_floor_only", "area_m2", 0.049),
        ("robust_floor_only", "max_vertex_residual_m", 0.021),
        ("robust_floor_plus_support", "area_m2", 0.009),
        ("robust_floor_plus_support", "weighted_p95_residual_m", 0.021),
        ("robust_floor_plus_support", "projected_aabb_fill", 0.249),
    ),
)
def test_robust_surface_contract_recomputes_selected_support_eligibility(
    policy: str, metric: str, value: float
) -> None:
    module = _load_runner()
    report = _collision(module, policy)
    support = report["raw_diagnostics"]["feature_extraction"]["robust_surfaces"][
        "components"
    ][1]
    support[metric] = value
    assert not module._robust_surface_contract(
        report, _surface_spec(policy), ["obj_00"]
    )


@pytest.mark.parametrize(
    "policy", ("robust_floor_only", "robust_floor_plus_support")
)
def test_robust_surface_contract_recomputes_unreported_support_eligibility(
    policy: str,
) -> None:
    module = _load_runner()
    report = _collision(module, policy)
    component = _append_box_false_overlap_component(report)
    component.update(
        box_footprint_safe=True,
        box_false_overlap_rejections=[],
        support_associations=[
            deepcopy(
                report["raw_diagnostics"]["feature_extraction"][
                    "robust_surfaces"
                ]["components"][1]["support_associations"][0]
            )
        ],
    )
    assert not module._robust_surface_contract(
        report, _surface_spec(policy), ["obj_00"]
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda report: report["raw_diagnostics"]["room_collision"][
            "robust_support_demotion"
        ].update(face_inventory_validated=False),
        lambda report: report["raw_diagnostics"]["room_collision"][
            "robust_support_demotion"
        ].update(residual_rebuild_face_count=26),
        lambda report: report["collision_exclusion"].update(
            unresolved_intrusion_count=1
        ),
        lambda report: report["raw_diagnostics"]["room_collision"][
            "support_rejections"
        ][0].update(max_penetration_depth_m=0.004),
    ),
)
def test_robust_surface_contract_rejects_invalid_support_demotion(mutator) -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    _reject_support(report)
    mutator(report)
    assert not module._robust_surface_contract(
        report,
        _surface_spec("robust_floor_plus_support"),
        ["obj_00"],
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda report: report.update(floor_z_m=0.0),
        lambda report: report["floor_patch_selection"].update(status="legacy"),
        lambda report: report["raw_diagnostics"]["feature_extraction"][
            "robust_surfaces"
        ]["guards"].update(support_min_area_m2=0.05),
        lambda report: report["raw_diagnostics"]["feature_extraction"][
            "robust_surfaces"
        ]["guards"].update(support_projected_overlap_method="aabb_overlap"),
        lambda report: report["raw_diagnostics"]["feature_extraction"][
            "robust_surfaces"
        ]["floor"].update(dominance_passed=False),
        lambda report: report["raw_diagnostics"]["dynamic_object_hulls"][0].pop(
            "floor_aware_clip"
        ),
    ),
)
def test_robust_surface_contract_fails_closed(mutator) -> None:
    module = _load_runner()
    report = _collision(module, "robust_floor_plus_support")
    mutator(report)
    assert not module._robust_surface_contract(
        report, _surface_spec("robust_floor_plus_support"), ["obj_00"]
    )


def _comparison(passed: bool, paired: int, *, errors: dict | None = None) -> dict:
    return {
        "comparison_pass": passed,
        "errors": {} if errors is None else errors,
        "paired_stable_count": paired,
    }


def test_selection_prefers_floor_unless_fs_strictly_improves() -> None:
    module = _load_runner()
    variants = {
        "3db-f": _comparison(True, 3),
        "3db-fs": _comparison(True, 3),
    }
    assert module._choose_scene_winner("3db0a1c8f3", variants)[0] == "3db-f"
    variants["3db-fs"] = _comparison(True, 4)
    assert module._choose_scene_winner("3db0a1c8f3", variants)[0] == "3db-fs"


def test_selection_uses_fs_only_when_floor_fails_or_strictly_improves() -> None:
    module = _load_runner()
    variants = {
        "d755-f": _comparison(False, 10),
        "d755-fs": _comparison(True, 2),
    }
    assert module._choose_scene_winner("d755b3d9d8", variants)[0] == "d755-fs"
    variants["d755-fs"] = _comparison(False, 10)
    assert module._choose_scene_winner("d755b3d9d8", variants)[0] is None


def test_diagnostic_flags_are_hard_coded_false() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for key in ("gpu_launch_allowed", "large_rollout_launch_allowed", "paper_ready"):
        assert f'"{key}": False' in source
    assert "sealed_65_run_controls_are_authenticated_references_only" in source


def _fake_aggregate(module, monkeypatch, comparisons: dict[str, dict]) -> dict:
    code = {
        "code_root": str(module.CODE_ROOT),
        "commit": "a" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(module._BASE, "_validated_id", lambda value: value)
    monkeypatch.setattr(module._BASE, "_code_snapshot", lambda _commit: code)
    monkeypatch.setattr(module._BASE, "_evidence_root", lambda: Path("/evidence"))
    monkeypatch.setattr(
        module._BASE,
        "_comparison_bundle",
        lambda _root, _sweep_id, variant_id: Path("/comparison") / variant_id,
    )
    monkeypatch.setattr(
        module,
        "_build_comparison_gate",
        lambda *, variant_id, **_kwargs: deepcopy(comparisons[variant_id]),
    )
    monkeypatch.setattr(
        module._BASE,
        "_validated_bundle_gate",
        lambda _directory, *, variant_id, **_kwargs: (
            deepcopy(comparisons[variant_id]),
            {"manifest_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        module._BASE,
        "_candidate",
        lambda: SimpleNamespace(_same_replay_structure=lambda left, right: left == right),
    )
    monkeypatch.setattr(
        module,
        "_control_reference",
        lambda _root, scene_id: {"scene_id": scene_id},
    )
    monkeypatch.setattr(
        module,
        "_publish",
        lambda _destination, **_kwargs: {"manifest_sha256": "c" * 64},
    )
    return module.aggregate(sweep_id="robust-test", expected_commit="a" * 40)


def test_aggregate_fails_closed_on_nested_comparison_runtime_error(
    monkeypatch,
) -> None:
    module = _load_runner()
    comparisons = {
        "3db-f": _comparison(
            False,
            0,
            errors={"A0": "missing sealed policy bundle"},
        ),
        "3db-fs": _comparison(True, 2),
        "d755-f": _comparison(True, 2),
        "d755-fs": _comparison(True, 3),
    }
    result = _fake_aggregate(module, monkeypatch, comparisons)
    assert result["status"] == "incomplete"
    assert result["comparison_error_variants"] == ["3db-f"]
    assert result["scene_winners"] == {
        "3db0a1c8f3": None,
        "d755b3d9d8": None,
    }
    assert {
        decision["reason"] for decision in result["scene_decisions"].values()
    } == {"comparison_artifact_or_runtime_error"}


def test_aggregate_remains_complete_for_scientific_failures_without_errors(
    monkeypatch,
) -> None:
    module = _load_runner()
    comparisons = {
        "3db-f": _comparison(False, 0),
        "3db-fs": _comparison(True, 2),
        "d755-f": _comparison(False, 0),
        "d755-fs": _comparison(False, 0),
    }
    result = _fake_aggregate(module, monkeypatch, comparisons)
    assert result["status"] == "complete"
    assert result["comparison_error_variants"] == []
    assert result["scene_winners"] == {
        "3db0a1c8f3": "3db-fs",
        "d755b3d9d8": None,
    }
    assert result["scene_decisions"]["3db0a1c8f3"]["reason"] == (
        "f_fails_and_fs_passes"
    )
    assert result["scene_decisions"]["d755b3d9d8"]["reason"] == (
        "neither_f_nor_fs_passes"
    )
