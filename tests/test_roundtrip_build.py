import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from robo.roundtrip.build import prepare_observations, read_train, target_prompt, validate_config


def seal(root, manifest):
    manifest["files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob("*")) if p.is_file() and p.name != "capture_manifest.json"}
    (root / "capture_manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    (root / "train/rgb").mkdir(parents=True)
    (root / "train/depth_m").mkdir()
    rows = []
    for i in range(2):
        frame = f"f{i:06d}"
        rgb = np.full((24, 32, 3), 100 + i, dtype=np.uint8)
        Image.fromarray(rgb).save(root / f"train/rgb/{frame}.png")
        depth = np.ones((24, 32), dtype=np.float32)
        depth[8:16, 12:20] = .8
        np.save(root / f"train/depth_m/{frame}.npy", depth)
        transform = np.eye(4)
        transform[0, 3] = .001 * i
        rows.append({"frame_id": frame, "rgb": f"train/rgb/{frame}.png", "depth_m": f"train/depth_m/{frame}.npy",
            "K": [[100, 0, 16], [0, 100, 12], [0, 0, 1]], "T_world_from_camera": transform.tolist()})
    (root / "train/cameras.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (root / "task_instruction.txt").write_text("Pick the red apple from the counter and place it in the sink.")
    (root / "robot_config.json").write_text("{}")
    seal(root, {"status": "PASS", "split": "train_only", "capture_id": "c-0123456789abcdef",
        "sensor_regime": "ideal_rgbd", "width": 32, "height": 24, "train_frames": 2})
    return root


@pytest.fixture
def config():
    return {"tier": "DEV", "scope": "target_only", "method": "B0_fixed_trellis", "seed": 42,
        "source_commit": "a" * 40, "minimum_mask_pixels": 32, "minimum_views": 2,
        "maximum_points_per_mask": 1000, "maximum_observation_points": 1000,
        "score_min": .45, "matching_fraction": .2, "matching_distance_m": .02}


def predictor(rgb, prompt):
    assert prompt == "red apple"
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    mask[8:16, 12:20] = True
    return [(mask, .9)]


def test_real_camera_depth_bridge_without_models(capture, config, tmp_path):
    out = tmp_path / "build"
    result = prepare_observations(capture, out, config, predictor)
    assert result["confirmed_views"] == 2
    assert result["read_splits"] == ["train"]
    assert result["observation_points"] == 128
    points = np.load(out / "discovery/observation_points.npy")
    np.testing.assert_allclose(points[:, 2], .8)
    np.testing.assert_allclose(points.min(0)[:2], [-.032, -.032], atol=1e-7)
    rgba = np.asarray(Image.open(out / "construction/objects/obj_00/rgba.png"))
    assert np.count_nonzero(rgba[..., 3]) == 64
    assert result["anchor_mask_pixel_fraction"] == 64 / (24 * 32)
    with pytest.raises(FileExistsError):
        prepare_observations(capture, out, config, predictor)


@pytest.mark.parametrize("mode", ["zero", "wrong_shape", "no_second_view"])
def test_bad_automatic_detection_never_falls_back(capture, config, tmp_path, mode):
    def bad(rgb, prompt):
        if mode == "zero":
            return []
        if mode == "wrong_shape":
            return [(np.ones((2, 3), bool), .9)]
        return predictor(rgb, prompt) if rgb[0, 0, 0] == 100 else []
    with pytest.raises(ValueError):
        prepare_observations(capture, tmp_path / "build", config, bad)
    assert not (tmp_path / "build/construction/objects/obj_00/rgba.png").exists()


def test_private_depth_reference_rejected_before_read(capture):
    cameras = capture / "train/cameras.jsonl"
    rows = [json.loads(line) for line in cameras.read_text().splitlines()]
    rows[0]["depth_m"] = "../../vault/native_depth.npy"
    cameras.write_text("\n".join(map(json.dumps, rows)))
    seal(capture, json.loads((capture / "capture_manifest.json").read_text()))
    with pytest.raises(ValueError, match="non-TRAIN"):
        read_train(capture)


def test_rgb_only_cannot_use_depth_bridge(capture):
    manifest = json.loads((capture / "capture_manifest.json").read_text())
    manifest["sensor_regime"] = "posed_rgb"
    seal(capture, manifest)
    with pytest.raises(ValueError):
        read_train(capture)


def test_instruction_is_only_target_source():
    assert target_prompt("Pick the blue mug from the table.", "mug") == "mug"
    with pytest.raises(ValueError):
        target_prompt("Pick the blue mug from the table.", "native_asset_123")


def test_unresolved_source_or_full_run_rejected(config):
    config["source_commit"] = "UNRESOLVED"
    with pytest.raises(ValueError):
        validate_config(config)
    config.update(source_commit="a" * 40, tier="TEST")
    with pytest.raises(ValueError):
        validate_config(config)


def test_v2_capture_identity_checked_before_masks(capture, config, tmp_path):
    config.update(schema_version=2, canonical_instance_id="instance-1", cohort_id="dev-eight",
                  capture_manifest_sha256=hashlib.sha256((capture / "capture_manifest.json").read_bytes()).hexdigest())
    result = prepare_observations(capture, tmp_path / "valid", config, predictor)
    assert result["canonical_instance_id"] == "instance-1"
    assert result["cohort_id"] == "dev-eight"
    config["capture_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="capture identity"):
        prepare_observations(capture, tmp_path / "invalid", config, predictor)
    assert not (tmp_path / "invalid/discovery").exists()


def test_v2_test_requires_dev_receipt(config):
    config.update(schema_version=2, canonical_instance_id="instance-1", cohort_id="test-core",
                  capture_manifest_sha256="a" * 64, tier="TEST")
    with pytest.raises(ValueError, match="DEV admission"):
        validate_config(config)
    config["dev_admission_sha256"] = "b" * 64
    validate_config(config)
    config["canonical_instance_id"] = ""
    with pytest.raises(ValueError, match="canonical_instance_id"):
        validate_config(config)


def test_public_destination_role_is_not_native_fixture_lookup(config):
    from robo.roundtrip.build import task_role_prompt
    instruction='Pick the lemon from the sink and place it on the plate located on the counter.'
    assert task_role_prompt(instruction)=='lemon'
    assert task_role_prompt(instruction,'receptacle')=='plate'
    with pytest.raises(ValueError,match='unresolved'):
        task_role_prompt('Pick the apple from the counter and place it in the sink.','receptacle')
    with pytest.raises(ValueError,match='differs'):
        task_role_prompt(instruction,'receptacle','native_container_asset_12')
    config.update(schema_version=2,canonical_instance_id='a',cohort_id='dev',capture_manifest_sha256='0'*64,
                  scope='role_asset',object_role='receptacle',declared_scope='L1_target_destination')
    validate_config(config)
    config['scope']='target_only'
    with pytest.raises(ValueError,match='silently'):validate_config(config)
