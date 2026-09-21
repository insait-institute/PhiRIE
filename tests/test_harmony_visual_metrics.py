import json

import numpy as np
from PIL import Image

from robo.eval.harmony_visual_metrics import mask_iou, deterministic_core_check


def test_mask_iou_on_identical_masks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 255
    Image.fromarray(mask).save(a / "frame.png")
    Image.fromarray(mask).save(b / "frame.png")
    value, count = mask_iou(a, b)
    assert count == 1
    assert value == 1.0


def test_core_check_rejects_any_nonzero_error(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({
        "treatment_id": "c",
        "per_camera_metadata": [{"cam": {
            "robot_core_equal": True, "max_robot_core_error": 0}}]}) + "\n")
    assert deterministic_core_check(path, "c") is True
    path.write_text(json.dumps({
        "treatment_id": "c",
        "per_camera_metadata": [{"cam": {
            "robot_core_equal": False, "max_robot_core_error": 1}}]}) + "\n")
    assert deterministic_core_check(path, "c") is False
