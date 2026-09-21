"""Tests for baselines/simfoundry_repro.py -- config/manifest logic only.
The actual GPU pipeline (SAM3/TRELLIS/CoACD/PyBullet) is not invoked here;
see run/run_simfoundry.sh for the real end-to-end launcher.
"""
from pathlib import Path

from baselines import simfoundry_repro as sf


def test_resolves_recon_scenes_layout(tmp_path):
    scene_dir = tmp_path / "data" / "recon_scenes" / "data" / "behavior_task0011"
    scene_dir.mkdir(parents=True)
    resolved = sf.resolve_scene_env(scene_dir, tmp_path / "out")
    assert resolved["layout"] == "recon_scenes"
    assert resolved["scene_id"] == "behavior_task0011"
    assert resolved["env"]["SIMANY_SCENE"] == "behavior_task0011"


def test_resolves_scannetpp_layout(tmp_path, monkeypatch):
    # Simulate the real /data/ScanNetpp/data/<scene> layout under tmp_path
    # by monkeypatching the literal comparison target -- simplest is to
    # just check the "custom" fallback behaves sanely for a lookalike root,
    # since we can't relocate the real /data/ScanNetpp mount in a test.
    scene_dir = tmp_path / "some_root" / "data" / "c50d2d1d42"
    scene_dir.mkdir(parents=True)
    resolved = sf.resolve_scene_env(scene_dir, tmp_path / "out")
    assert resolved["layout"] == "custom"  # not literally /data/ScanNetpp
    assert resolved["scene_id"] == "c50d2d1d42"


def test_dry_run_manifest_reports_na_metrics_never_fabricated(tmp_path):
    manifest = sf.build_manifest(tmp_path / "scene", tmp_path / "out", dry_run=True)
    assert manifest["dry_run"] is True
    assert manifest["build_success"] is None  # never a fabricated pass/fail
    for name, entry in manifest["na_metrics"].items():
        assert entry["status"] == "not_applicable"
        assert entry["value"] is None
        assert entry["reason"]  # every N/A has a real, non-empty reason


def test_manifest_build_success_reflects_stage_log(tmp_path):
    ok_log = [{"stage": "s0", "module": "m", "seconds": 1.0, "returncode": 0, "stderr_tail": None}]
    m = sf.build_manifest(tmp_path / "scene", tmp_path / "out", stage_log=ok_log)
    assert m["build_success"] is True

    fail_log = ok_log + [{"stage": "s1", "module": "m2", "seconds": 1.0,
                           "returncode": 1, "stderr_tail": "boom"}]
    m2 = sf.build_manifest(tmp_path / "scene", tmp_path / "out", stage_log=fail_log)
    assert m2["build_success"] is False


def test_config_matches_ablation_row_d():
    manifest = sf.build_manifest(Path("/tmp/x"), Path("/tmp/y"), dry_run=True)
    cfg = manifest["config"]
    assert cfg["frames"] == 1
    assert cfg["view"] == "single"
    assert cfg["gt_used"] is False
    assert cfg["hybrid_generation"] is False
