import pytest

from robo.eval.audit_labels import ValidityThresholds, label_record
from robo.eval.audit_loso import SIGNAL_GROUP_NAMES, loso_predictions, signal_groups


def test_missing_required_evidence_is_invalid():
    record = {
        "object_recovered": True, "translation_cm": 1,
        "rotation_deg": 1, "scale_error_pct": 1,
        "penetration_cm": 0, "support_correct": True,
        "collision_valid": True,
    }
    result = label_record(record, ValidityThresholds())
    assert result["invalid_label"] == 1
    assert "missing_required_visible" in result["invalid_reasons"]


def test_loso_outputs_each_signal_for_each_query():
    rows = []
    for scene_index in range(3):
        for query_index in range(4):
            invalid = int(query_index >= 2)
            rows.append({
                "scene_id": f"s{scene_index}", "task_id": f"q{query_index}",
                "invalid_label": invalid,
                "global_psnr_db": 30.0 - 10.0 * invalid,
                "global_f1_score": 0.9 - 0.6 * invalid,
                "scene_build_score": 0.1 + 0.7 * invalid,
                "visual_error": 0.1 + 0.7 * invalid,
                "geometry_error": 0.1 + 0.7 * invalid,
                "support_error": 0.1 + 0.7 * invalid,
                "robot_error": 0.1 + 0.7 * invalid,
            })
    predictions, groups = loso_predictions(rows)
    assert tuple(groups) == SIGNAL_GROUP_NAMES
    assert len(predictions) == len(rows) * len(groups)
    assert {row["heldout_scene"] for row in predictions} == {"s0", "s1", "s2"}
    assert all(0.0 <= row["invalid_probability"] <= 1.0 for row in predictions)


def test_signal_schema_rejects_stale_global_lpips_row():
    columns = [
        "global_psnr_db", "global_f1_score", "global_lpips_score",
        "scene_score", "visual_score", "geometry_score", "support_score",
        "robot_score",
    ]
    with pytest.raises(ValueError, match="stale Global LPIPS"):
        signal_groups(columns)


def test_signal_schema_requires_all_six_rows():
    with pytest.raises(ValueError, match="Global geometry F1 only"):
        signal_groups([
            "global_psnr_db", "scene_score", "visual_score",
            "geometry_score", "support_score", "robot_score",
        ])
