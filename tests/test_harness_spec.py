import pytest

from robo.eval.harness_spec import load_harness_spec


def test_single_axis_comparisons_validate():
    spec = load_harness_spec({
        "treatments": [
            {"id": "proxy", "scene": "box_proxy"},
            {"id": "sim", "scene": "simany"},
            {"id": "raw", "scene": "simany", "observation": "composite_raw"}],
        "comparisons": [
            {"id": "scene", "axis": "scene", "treatments": ["proxy", "sim"]},
            {"id": "obs", "axis": "observation", "treatments": ["sim", "raw"]}]})
    assert spec.treatment_ids() == ("proxy", "sim", "raw")


def test_mixed_axis_comparison_is_rejected():
    with pytest.raises(ValueError, match="changes"):
        load_harness_spec({
            "treatments": [
                {"id": "a", "scene": "box_proxy", "observation": "raster"},
                {"id": "b", "scene": "simany", "observation": "composite_raw"}],
            "comparisons": [
                {"id": "bad", "axis": "scene", "treatments": ["a", "b"]}]})


def test_e3_construction_policy_names_are_explicit_scene_variants():
    contract = {
        "policy": {"id": "scripted_sinusoid", "kind": "scripted_smoke"},
        "robot": {"id": "fixture"},
        "cameras": {"id": "fixture"},
        "action_convention": "absolute_joint_position",
        "controller": {"id": "fixture"},
        "control_rate_hz": 15,
        "horizon_s": 1,
        "task_instruction": "fixture",
        "rubric": {"id": "fixture"},
        "reset_ids": ["fixture-reset"],
    }
    spec = load_harness_spec({
        "policy": "scripted_sinusoid",
        "contract": contract,
        "treatments": [
            {"id": "a0", "scene": "fixed_single_path"},
            {"id": "a4", "scene": "agentic"},
        ],
        "comparisons": [{
            "id": "construction", "axis": "scene",
            "treatments": ["a0", "a4"],
        }],
    })
    assert spec.treatments["a0"].scene == "fixed_single_path"
    assert spec.treatments["a4"].scene == "agentic"


def test_e3_construction_variants_require_frozen_contract():
    with pytest.raises(ValueError, match="require an explicit global frozen contract"):
        load_harness_spec({
            "treatments": [
                {"id": "a0", "scene": "fixed_single_path"},
                {"id": "a4", "scene": "agentic"},
            ],
            "comparisons": [{
                "id": "construction", "axis": "scene",
                "treatments": ["a0", "a4"],
            }],
        })


def test_e3_construction_variants_cannot_drift_in_collision_comparison():
    with pytest.raises(ValueError, match="changes.*scene"):
        load_harness_spec({
            "treatments": [
                {"id": "a0", "scene": "fixed_single_path"},
                {"id": "a4", "scene": "agentic",
                 "collision": "private_shims"},
            ],
            "comparisons": [{
                "id": "collision", "axis": "collision",
                "treatments": ["a0", "a4"],
            }],
        })


def test_e4_runtime_policy_and_rate_must_match_declared_contract():
    base = {
        "policy": "scripted_sinusoid",
        "contract": {
            "policy": {"id": "scripted_sinusoid", "kind": "scripted_smoke"},
            "robot": {"id": "fixture"},
            "cameras": {"id": "fixture"},
            "action_convention": "absolute_joint_position",
            "controller": {"id": "fixture"},
            "control_rate_hz": 15,
            "horizon_s": 1,
            "task_instruction": "fixture",
            "rubric": {"id": "fixture"},
            "reset_ids": ["reset"],
        },
        "treatments": [
            {"id": "a0", "scene": "fixed_single_path"},
            {"id": "a4", "scene": "agentic"},
        ],
        "comparisons": [
            {"id": "construction", "axis": "scene", "treatments": ["a0", "a4"]}
        ],
    }
    wrong_policy = dict(base)
    wrong_policy["policy"] = "different-runtime-policy"
    with pytest.raises(ValueError, match="top-level policy differs"):
        load_harness_spec(wrong_policy)

    wrong_rate = {**base, "contract": {**base["contract"], "control_rate_hz": 99}}
    with pytest.raises(ValueError, match="control_rate_hz differs"):
        load_harness_spec(wrong_rate)
