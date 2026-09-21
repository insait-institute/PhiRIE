import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from robo.eval import fidelity_metrics
from robo.eval.fidelity_metrics import geometry_metrics, load_surface, psnr, ssim


def test_identical_images_have_perfect_pixel_metrics():
    image = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert psnr(image, image) == float("inf")
    assert abs(ssim(image, image) - 1.0) < 1e-10


def test_ssim_agrees_with_frozen_trusted_skimage_reference():
    rng = np.random.default_rng(123)
    target = rng.random((32, 32, 3))
    prediction = np.clip(target * 0.83 + 0.07, 0, 1)
    # skimage.metrics.structural_similarity(..., channel_axis=2,
    # data_range=1.0), independently evaluated in the MVPY environment.
    trusted = 0.9825222840560377
    assert ssim(prediction, target) == pytest.approx(trusted, abs=1e-12)


def test_geometry_metrics_are_metric_and_symmetric():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    result = geometry_metrics(points, points.copy())
    assert result["cd_cm"] == 0.0
    assert result["f1_20"] == 1.0
    assert not result["collapse"]


def test_mask_restricted_metrics_ignore_pixels_outside_mask():
    target = np.zeros((24, 24, 3), dtype=np.float32)
    pred_a = target.copy()
    pred_b = target.copy()
    pred_a[0, 0] = 1
    pred_b[:8, :8] = 1
    mask = np.zeros((24, 24), dtype=bool)
    mask[12:20, 12:20] = True
    assert psnr(pred_a, target, mask) == psnr(pred_b, target, mask)
    assert ssim(pred_a, target, mask) == pytest.approx(ssim(pred_b, target, mask))


def test_collapse_threshold_is_strictly_below_point_one():
    shared = np.array([[0.0, 0.0, 0.0]])
    pred = np.concatenate([shared, np.arange(10.0, 19.0)[:, None] * np.ones((1, 3))])
    target = np.concatenate([shared, -np.arange(10.0, 19.0)[:, None] * np.ones((1, 3))])
    result = geometry_metrics(pred, target)
    assert result["f1_20"] == pytest.approx(0.1)
    assert result["collapse"] is False

    pred = np.concatenate([shared, np.arange(10.0, 29.0)[:, None] * np.ones((1, 3))])
    target = np.concatenate([shared, -np.arange(10.0, 29.0)[:, None] * np.ones((1, 3))])
    assert geometry_metrics(pred, target)["collapse"] is True


class _FakeLPIPS:
    error = None
    model = object()
    provenance = {
        "backend_class": "tests.FakeLPIPS",
        "package": "test",
        "package_version": "1",
        "network": "alex",
        "backbone_checkpoint": {"path": "fake", "size_bytes": 1, "sha256": "a" * 64},
        "linear_checkpoint": {"path": "fake", "size_bytes": 1, "sha256": "b" * 64},
    }

    def __init__(self, _device="cpu"):
        pass

    def __call__(self, _pred, _target, _mask=None):
        return 0.25


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path, value, *, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 16), dtype=np.uint8) if mask else np.zeros((16, 16, 3), dtype=np.uint8)
    if mask:
        array[4:12, 4:12] = value
    else:
        array[4:12, 4:12] = value
    Image.fromarray(array).save(path)


def _synthetic_manifest(root):
    room_render = root / "room-render"
    room_gt = root / "room-gt"
    object_render = root / "object-render"
    object_gt = root / "object-gt"
    masks = root / "masks"
    _write_image(room_render / "frame.png", 64)
    _write_image(room_gt / "frame.png", 32)
    _write_image(object_render / "heldout.png", 80)
    _write_image(object_gt / "heldout.png", 40)
    _write_image(masks / "heldout.png", 255, mask=True)
    pred_surface = root / "pred.npy"
    gt_surface = root / "gt.npy"
    registration = root / "registration.npy"
    np.save(pred_surface, np.array([[0, 0, 0], [0.01, 0, 0]], dtype=float))
    np.save(gt_surface, np.array([[0, 0, 0], [0.02, 0, 0]], dtype=float))
    np.save(registration, np.array([[1, 1, 1], [2, 2, 2]], dtype=float))
    return {
        "schema_version": 1,
        "freeze_id": "freeze-test",
        "paper_ready": False,
        "room_split": {"unit": "held_out_view"},
        "object_split": {
            "f1_threshold_mm": 20,
            "catastrophic_collapse_below": 0.1,
            "require_independent_registration_surface": True,
        },
        "coverage": {
            "minimum_room_views_by_method": {"room-id": 1},
            "minimum_objects_by_method": {"object-id": 1},
        },
        "room_methods": {
            "room": {
                "source_id": "room-id",
                "records": [{
                    "freeze_id": "freeze-test",
                    "scene_id": "scene-a",
                    "source_build": "build-a",
                    "render_dir": str(room_render),
                    "gt_dir": str(room_gt),
                    "coverage": {"render_only": [], "gt_only": []},
                    "n_views": 1,
                    "evaluation_frames": ["frame.png"],
                    "optimization_input_frames": ["train.png"],
                    "views": [{
                        "view_id": "frame.JPG",
                        "render_path": str(room_render / "frame.png"),
                        "gt_path": str(room_gt / "frame.png"),
                        "render_sha256": _sha(room_render / "frame.png"),
                        "gt_sha256": _sha(room_gt / "frame.png"),
                    }],
                }],
            }
        },
        "object_methods": {
            "object": {
                "source_id": "object-id",
                "records": [{
                    "freeze_id": "freeze-test",
                    "scene_id": "scene-a",
                    "object_id": "object-a",
                    "category_id": "chair",
                    "source_build": "build-a",
                    "render_dir": str(object_render),
                    "gt_dir": str(object_gt),
                    "mask_dir": str(masks),
                    "coverage": {"render_only": [], "gt_only": []},
                    "evaluation_frames": ["heldout.png"],
                    "generator_input_frames": ["input.png"],
                    "pred_surface": str(pred_surface),
                    "gt_surface": str(gt_surface),
                    "registration_input_surface": str(registration),
                    "registration_input_surface_hash": _sha(registration),
                    "evaluation_surface_hash": _sha(gt_surface),
                    "units": "m",
                    "coordinate_frame": "scene-a-world",
                    "alignment_convention": "pred_to_gt_rigid",
                    "surface_sample_count": 2,
                }],
            }
        },
    }


def _evaluate(tmp_path, monkeypatch, manifest=None, name="table"):
    monkeypatch.setattr(fidelity_metrics, "LPIPSEvaluator", _FakeLPIPS)
    manifest = manifest or _synthetic_manifest(tmp_path)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest))
    return fidelity_metrics.evaluate_manifest(
        path, tmp_path / name, bootstrap_samples=32, bootstrap_seed=7
    )


def test_tiny_strict_manifest_has_counts_hashes_bootstrap_and_atomic_outputs(tmp_path, monkeypatch):
    payload = _evaluate(tmp_path, monkeypatch)
    assert payload["validation"]["valid"] is True
    assert payload["validation"]["lpips_provenance_complete"] is True
    assert payload["lpips_provenance"]["network"] == "alex"
    assert payload["paper_ready"] is False
    assert payload["rows_count"] == 2
    assert {path.name for path in (tmp_path / "table").iterdir()} == {
        "fidelity_table.json", "fidelity_table.csv"}
    for row in payload["rows"]:
        assert row["metric_samples"]
        assert row["metric_source_hashes"]
        for detail in row["metrics"].values():
            assert detail["n"] == 1
            assert len(detail["source_artifact_hash"]) == 64
            assert detail["bootstrap"]["ci95"][0] is not None
    persisted = json.loads((tmp_path / "table" / "fidelity_table.json").read_text())
    assert persisted == payload


def test_output_directory_is_never_overwritten(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    _evaluate(tmp_path, monkeypatch, manifest)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_mismatched_filenames_require_exact_coverage(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    render_dir = next(iter(manifest["room_methods"].values()))["records"][0]["render_dir"]
    _write_image(__import__("pathlib").Path(render_dir) / "extra.png", 1)
    with pytest.raises(ValueError, match="without an exact coverage record"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_generator_frame_leakage_fails(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    record = manifest["object_methods"]["object"]["records"][0]
    record["generator_input_frames"] = ["heldout.png"]
    with pytest.raises(ValueError, match="generator/evaluation frame leakage"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_room_optimization_input_frames_are_required(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    del manifest["room_methods"]["room"]["records"][0][
        "optimization_input_frames"
    ]
    with pytest.raises(ValueError, match="missing required optimization_input_frames"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_room_view_hash_drift_fails(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    view = manifest["room_methods"]["room"]["records"][0]["views"][0]
    view["render_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="render_sha256 does not match"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_identical_registration_and_evaluation_hash_fails(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    record = manifest["object_methods"]["object"]["records"][0]
    record.pop("registration_input_surface")
    record["registration_input_surface_hash"] = record["evaluation_surface_hash"]
    with pytest.raises(ValueError, match="hashes are identical"):
        _evaluate(tmp_path, monkeypatch, manifest)


@pytest.mark.parametrize("field", ["units", "coordinate_frame", "alignment_convention", "surface_sample_count"])
def test_geometry_contract_fields_are_required(tmp_path, monkeypatch, field):
    manifest = _synthetic_manifest(tmp_path)
    del manifest["object_methods"]["object"]["records"][0][field]
    with pytest.raises(ValueError, match=field):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_empty_mask_and_surface_fail_loudly(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    record = manifest["object_methods"]["object"]["records"][0]
    _write_image(__import__("pathlib").Path(record["mask_dir"]) / "heldout.png", 0, mask=True)
    with pytest.raises(ValueError, match="empty evaluation mask"):
        _evaluate(tmp_path, monkeypatch, manifest)

    empty = tmp_path / "empty.npy"
    np.save(empty, np.empty((0, 3)))
    with pytest.raises(ValueError, match="empty point set"):
        load_surface(empty)


def test_manifest_rejects_artifact_path_outside_repository(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    manifest["room_methods"]["room"]["records"][0]["render_dir"] = "/outside-simany/render"
    with pytest.raises(ValueError, match="outside the SimAny repository"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_manifest_rejects_symlink_components(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path)
    actual = tmp_path / "room-render"
    link = tmp_path / "linked-render"
    link.symlink_to(actual, target_is_directory=True)
    manifest["room_methods"]["room"]["records"][0]["render_dir"] = str(link)
    with pytest.raises(ValueError, match="symlink component"):
        _evaluate(tmp_path, monkeypatch, manifest)


def test_manifest_and_output_paths_are_repository_confined(tmp_path, monkeypatch):
    monkeypatch.setattr(fidelity_metrics, "LPIPSEvaluator", _FakeLPIPS)
    with pytest.raises(ValueError, match="outside the SimAny repository"):
        fidelity_metrics.evaluate_manifest(
            "/outside-simany/manifest.json", tmp_path / "table"
        )
    manifest = _synthetic_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside the SimAny repository"):
        fidelity_metrics.evaluate_manifest(
            manifest_path, "/outside-simany/table", bootstrap_samples=8
        )
