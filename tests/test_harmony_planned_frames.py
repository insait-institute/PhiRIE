import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robo.eval import harmony_visual_metrics as h


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "frames"
    root.mkdir()
    raw, gt, core = root / "raw", root / "gt", root / "core"
    for directory in (raw, gt, core): directory.mkdir()
    frames = []
    for index in range(2):
        name = f"{index}.png"
        rgb = np.full((16, 16, 3), 100 + index, dtype=np.uint8)
        Image.fromarray(rgb).save(raw / name)
        Image.fromarray(rgb + 10).save(gt / name)
        mask = np.zeros((16, 16), dtype=np.uint8); mask[5:10, 5:10] = 255
        Image.fromarray(mask).save(core / name)
        frames.append(dict(filename=name, run_id="run", scene_id="scene", task_id="task",
                           reset_id="reset", camera_id="wrist", frame_index=index,
                           raw=ref(raw / name), gt=ref(gt / name), robot_core_mask=ref(core / name)))
    conditions = {}
    for name in h.REQUIRED_CONDITIONS:
        directory = root / name; directory.mkdir()
        records = []
        for frame in frames:
            path = directory / frame["filename"]
            path.write_bytes(Path(frame["raw"]["path"]).read_bytes())
            records.append(dict(filename=frame["filename"], status="success", render=ref(path), end_to_end_ms=10))
        conditions[name] = dict(render_dir=str(directory), gt_dir=str(gt), frame_records=records)
    return dict(schema_version=2, planned_frames=frames, conditions=conditions)


def test_failed_enhancement_retains_denominator_and_attempt_latency(manifest):
    spec = manifest["conditions"]["harmonizer_temporal"]
    failed = spec["frame_records"][1]
    Path(failed.pop("render")["path"]).unlink()
    failed.update(status="enhancer_failure", failure_reason="timeout", end_to_end_ms=100)
    row = h.validate_planned_frames(manifest)["harmonizer_temporal"]
    assert (row["processed_frames"], row["planned_frames"], row["enhancer_failures"]) == (1, 2, 1)
    assert row["frame_coverage"] == .5 and row["p95_ms_per_camera"] == 95.5
    assert row["latency_samples"] == 2


def test_core_checks_pixels_instead_of_asserted_success(manifest):
    spec = manifest["conditions"]["harmonizer_robot_restore_c"]
    record = spec["frame_records"][0]; path = Path(record["render"]["path"])
    with Image.open(path) as im: rgb = np.array(im)
    rgb[6, 6, 0] += 1; Image.fromarray(rgb).save(path); record["render"] = ref(path)
    record["robot_core_equal"] = True
    result = h.validate_planned_frames(manifest)["harmonizer_robot_restore_c"]
    assert result["robot_core_exact_pixels"] is False
    assert result["robot_core_checked_frames"] == 2


@pytest.mark.parametrize("mutation,error", [
    ("missing_condition", "five declared"), ("duplicate_identity", "duplicate planned"),
    ("missing_record", "terminal records"), ("fallback", "render filenames"),
    ("tamper", "frame hash"), ("shape", "shape mismatch"),
    ("bad_latency", "finite nonnegative"), ("extra_image", "render filenames"),
    ("mask_names", "filenames differ"), ("empty_core", "nonempty"),
    ("unknown_schema", "unsupported visual"),
    ("missing_evaluation_masks", "evaluation masks differ"),
])
def test_invalid_manifest_fails_before_metrics(manifest, mutation, error, tmp_path, monkeypatch):
    spec = manifest["conditions"]["harmonizer_robot_restore_c"]
    record = spec["frame_records"][0]; path = Path(record["render"]["path"])
    if mutation == "missing_condition": del manifest["conditions"]["color_match"]
    elif mutation == "duplicate_identity": manifest["planned_frames"][1]["frame_index"] = 0
    elif mutation == "missing_record": spec["frame_records"].pop()
    elif mutation == "fallback": record.update(status="enhancer_failure", failure_reason="timeout")
    elif mutation == "tamper": path.write_bytes(b"changed")
    elif mutation == "shape":
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path); record["render"] = ref(path)
    elif mutation == "bad_latency": record["end_to_end_ms"] = float("nan")
    elif mutation == "extra_image": (path.parent / "extra.png").write_bytes(path.read_bytes())
    elif mutation == "mask_names":
        spec["robot_reference_mask_dir"] = spec["render_dir"]
        missing = tmp_path / "missing_masks"; missing.mkdir()
        spec["robot_prediction_mask_dir"] = str(missing)
    elif mutation == "empty_core":
        frame = manifest["planned_frames"][0]; mask = Path(frame["robot_core_mask"]["path"])
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(mask); frame["robot_core_mask"] = ref(mask)
    elif mutation == "unknown_schema": manifest["schema_version"] = 3
    elif mutation == "missing_evaluation_masks": spec["evaluation_mask_dir"] = str(tmp_path / "absent")
    def forbidden(*args): raise AssertionError("model initialization preceded manifest validation")
    monkeypatch.setattr(h, "LPIPSEvaluator", forbidden)
    source = tmp_path / "manifest.json"; source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error): h.generate(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_unattempted_frames_remain_null_and_no_output_reuse(manifest, tmp_path, monkeypatch):
    for spec in manifest["conditions"].values():
        for record in spec["frame_records"]:
            Path(record.pop("render")["path"]).unlink()
            record.update(status="not_run", failure_reason="model unavailable", end_to_end_ms=None)
    rows = h.validate_planned_frames(manifest)
    assert all(r["planned_frames"] == 2 and r["processed_frames"] == 0 and r["p95_ms_per_camera"] is None for r in rows.values())
    source = tmp_path / "manifest.json"; source.write_text(json.dumps(manifest))
    out = tmp_path / "out"; out.mkdir()
    monkeypatch.setattr(h, "LPIPSEvaluator", lambda *args: pytest.fail("model loaded"))
    with pytest.raises(FileExistsError): h.generate(source, out)


def test_existing_metric_producer_emits_coverage_before_quality(manifest, tmp_path, monkeypatch):
    class UnavailableLPIPS:
        model = None
        error = "synthetic test: LPIPS not loaded"
        def __init__(self, device): pass
        def __call__(self, a, b, mask=None): return None
    monkeypatch.setattr(h, "LPIPSEvaluator", UnavailableLPIPS)
    source = tmp_path / "manifest.json"; source.write_text(json.dumps(manifest))
    out = tmp_path / "result"
    result = h.generate(source, out)
    assert len(result["rows"]) == 5
    assert all(r["n_images"] == r["processed_frames"] == r["planned_frames"] == 2 for r in result["rows"])
    assert all(r["psnr"] > 0 and r["ssim"] > 0 and r["lpips"] is None for r in result["rows"])
    option_c = next(r for r in result["rows"] if r["condition"] == "harmonizer_robot_restore_c")
    assert option_c["robot_core_exact"] is True
    latex = (out / "harmony_visual_table.tex").read_text()
    assert latex.index("Processed/planned") < latex.index("PSNR $\\uparrow$")
    assert "2/2" in latex
    assert json.loads((out / "harmony_visual_table.json").read_text())["rows"] == result["rows"]
    assert (out / "input_manifest.json").read_bytes() == source.read_bytes()


def test_failed_publication_leaves_no_partial_result(manifest, tmp_path, monkeypatch):
    class UnavailableLPIPS:
        model = None
        error = "synthetic test"
        def __init__(self, device): pass
        def __call__(self, a, b, mask=None): return None
    monkeypatch.setattr(h, "LPIPSEvaluator", UnavailableLPIPS)
    def fail(*args): raise OSError("injected write failure")
    monkeypatch.setattr(h, "write_csv", fail)
    source = tmp_path / "manifest.json"; source.write_text(json.dumps(manifest))
    out = tmp_path / "unpublished"
    with pytest.raises(OSError, match="injected write failure"): h.generate(source, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".unpublished.staging-*"))
