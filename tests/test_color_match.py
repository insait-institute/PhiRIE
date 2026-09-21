import copy
import numpy as np
import pytest
from robo.rendering.color_match import fit_background_affine, apply_background_affine
from robo.eval.observation_pipeline import ObservationPipeline, ObservationError, ObservationSource


def fit(x, y, mask=None, **kwargs):
    return fit_background_affine(x, y, np.ones(x.shape[:2], dtype=bool) if mask is None else mask,
        camera_id="wrist", calibration_view_id="train0", construction_view_ids=["train0"],
        evaluation_view_ids=kwargs.get("evaluation_view_ids", ["test0"]))


def test_recovers_declared_affine_and_is_deterministic():
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    target = 2 * source + 10
    calibration = fit(source, target)
    assert np.array_equal(apply_background_affine(source, calibration, camera_id="wrist"), target)
    assert fit(source, target) == calibration
    assert np.array_equal(source, np.arange(48, dtype=np.uint8).reshape(4, 4, 3))


def test_foreground_does_not_enter_fit():
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    target = source + 10
    mask = np.ones((4, 4), dtype=bool); mask[0] = False
    before = fit(source, target, mask)
    target[0] = 255; source[0] = 0
    after = fit(source, target, mask)
    assert before["gain"] == after["gain"] and before["offset"] == after["offset"]
    assert before["calibration_sha256"] != after["calibration_sha256"]


def test_constant_channel_has_declared_gain_one():
    source = np.full((3, 4, 3), 50, dtype=np.uint8)
    target = np.full_like(source, 100)
    calibration = fit(source, target)
    assert calibration["gain"] == [1., 1., 1.]
    assert np.array_equal(apply_background_affine(source, calibration, camera_id="wrist"), target)


def test_split_overlap_and_empty_mask_rejected():
    source = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError, match="overlaps"): fit(source, source, evaluation_view_ids=["train0"])
    with pytest.raises(ValueError, match="nonempty"): fit(source, source, np.zeros((4, 4), bool))
    with pytest.raises(ValueError, match="uint8"): fit(source.astype(float), source)


def test_changed_calibration_or_camera_is_rejected():
    source = np.zeros((4, 4, 3), np.uint8); calibration = fit(source, source)
    with pytest.raises(ValueError, match="camera differs"):
        apply_background_affine(source, calibration, camera_id="exterior")
    changed = copy.deepcopy(calibration); changed["offset"][0] = 1
    with pytest.raises(ValueError, match="hash/camera differs"):
        apply_background_affine(source, changed, camera_id="wrist")


def test_pipeline_changes_only_rgb_and_never_refits():
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    qpos = np.arange(7, dtype=float)
    class Source(ObservationSource):
        def get_obs(self): return {"wrist": image, "qpos": qpos, "contacts": [(1, 2)]}
    calibration = fit(image, image + 10)
    pipeline = ObservationPipeline(Source(), variant="color_match", env=None,
        image_keys=["wrist"], color_match_calibrations={"wrist": calibration})
    pipeline.reset("episode1")
    calibration["offset"][0] = 200  # Caller mutation cannot change frozen pipeline parameters.
    first = pipeline.get_obs(); pipeline.reset("episode2"); second = pipeline.get_obs()
    assert np.array_equal(first.observation["wrist"], image + 10)
    assert np.array_equal(first.observation["wrist"], second.observation["wrist"])
    assert first.observation["qpos"] is qpos and first.observation["contacts"] == [(1, 2)]
    assert first.per_camera["wrist"]["backend"] == "color_match"
    with pytest.raises(ObservationError, match="each camera"):
        ObservationPipeline(Source(), variant="color_match", env=None, image_keys=["wrist"])
