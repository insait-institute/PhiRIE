"""Tests for robo.manifest (Task 01, plan/01_EXPERIMENT_MANIFEST.md).

Run with:
    .venv/bin/python -m pytest -q tests/test_manifest_roundtrip.py
"""
import json
from pathlib import Path

import pytest
import yaml

from robo.manifest import io as manifest_io
from robo.manifest import hash as manifest_hash
from robo.manifest.hash import (
    canonical_hash,
    find_semantic_issues,
    hash_checkpoint_path,
    manifest_content_hash,
)
from robo.manifest.schema import (
    ObservationPreprocessing,
    RolloutManifest,
    SceneBuildManifest,
    StagedProgress,
)
from robo.manifest.schema import (
    CaptureInputs,
    ScaleAlignmentResiduals,
    SimulatorVersions,
    VerificationResults,
)


def make_rollout_manifest(**overrides) -> RolloutManifest:
    fields = dict(
        created_utc="2026-08-16T00:00:00Z",
        git_dirty=False,
        scene_build_commit="d0ba6fe",
        scene_manifest_hash="a" * 64,
        policy_checkpoint_hash="b" * 64,
        controller_config_hash="c" * 64,
        camera_config_hash="d" * 64,
        task_id="c50d2d1d42_factory__obj_09_to_region",
        initial_state_id="c50d2d1d42_factory__obj_09_to_region__ep0",
        rollout_seed=0,
        scene_id="c50d2d1d42_factory",
        language="move the mug to the right side of the table",
        rubric_version="staged_0.25_per_stage_v1",
        horizon_s=16.0,
        observation_preprocessing=ObservationPreprocessing(mode="raster"),
        action_convention="absolute_joint_position",
        action_dim=8,
        video_path="runs/pi05_c50/ep0.mp4",
        success=True,
        staged_progress=StagedProgress(grasp=True, lift=True, hover=True, place=True),
    )
    fields.update(overrides)
    return RolloutManifest(**fields)


def make_build_manifest(**overrides) -> SceneBuildManifest:
    fields = dict(
        created_utc="2026-08-16T00:00:00Z",
        git_dirty=False,
        scene_build_commit="d0ba6fe",
        scene_manifest_hash="a" * 64,
        scene_id="c50d2d1d42_factory",
        capture_inputs=CaptureInputs(source_type="scannetpp_scan",
                                      source_path="c50d2d1d42/dslr"),
        scale_alignment_residuals=ScaleAlignmentResiduals(
            rotation_deg=0.1, translation_m=0.01, scale_ratio=1.001, rmse_m=0.02),
        full_room_collision_hash="e" * 64,
        simulator_versions=SimulatorVersions(python="3.11.9", mujoco="3.2.0"),
        verification_results=VerificationResults(passed=True),
    )
    fields.update(overrides)
    return SceneBuildManifest(**fields)


def test_git_snapshot_records_full_commit(monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"

    def fake_check_output(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return commit + "\n"
        if command[-2:] == ["status", "--porcelain"]:
            return ""
        if command[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "agent/test\n"
        raise AssertionError(command)

    monkeypatch.setattr(manifest_hash.subprocess, "check_output", fake_check_output)
    assert manifest_hash.git_snapshot("/repo") == {
        "commit": commit,
        "dirty": False,
        "branch": "agent/test",
    }


# --------------------------------------------------------- round-trip -----

def test_yaml_json_roundtrip_equal(tmp_path):
    m = make_rollout_manifest()
    json_path = tmp_path / "manifest.json"
    yaml_path = tmp_path / "manifest.yaml"
    manifest_io.dump_json(m, json_path)
    manifest_io.dump_yaml(m, yaml_path)

    m_from_json = manifest_io.load_json(json_path)
    m_from_yaml = manifest_io.load_yaml(yaml_path)

    assert m_from_json == m
    assert m_from_yaml == m
    assert manifest_io.to_dict(m_from_json) == manifest_io.to_dict(m_from_yaml)
    assert manifest_content_hash(m_from_json) == manifest_content_hash(m_from_yaml)
    assert manifest_content_hash(m_from_json) == manifest_content_hash(m)


def test_load_any_dispatches_on_suffix(tmp_path):
    m = make_build_manifest()
    manifest_io.dump_json(m, tmp_path / "b.json")
    manifest_io.dump_yaml(m, tmp_path / "b.yaml")
    assert manifest_io.load_any(tmp_path / "b.json") == m
    assert manifest_io.load_any(tmp_path / "b.yaml") == m


# ------------------------------------------------------- key-order hash ---

def test_key_order_does_not_change_hash():
    d1 = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2, 3]}
    d2 = {"c": [1, 2, 3], "b": {"y": 2, "x": 1}, "a": 1}
    assert canonical_hash(d1) == canonical_hash(d2)


def test_manifest_dict_dump_key_order_does_not_change_content_hash():
    m = make_rollout_manifest()
    d_normal = manifest_io.to_dict(m)
    # Rebuild the dict with keys inserted in reverse order -- semantically
    # identical, structurally shuffled.
    d_shuffled = {k: d_normal[k] for k in reversed(list(d_normal))}
    assert canonical_hash(d_normal) == canonical_hash(d_shuffled)


def test_reserialized_manifest_same_hash_regardless_of_yaml_or_json_path(tmp_path):
    m = make_rollout_manifest()
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.yaml"
    manifest_io.dump_json(m, p1)
    manifest_io.dump_yaml(m, p2)
    h1 = manifest_content_hash(manifest_io.load_json(p1))
    h2 = manifest_content_hash(manifest_io.load_yaml(p2))
    assert h1 == h2


# ------------------------------------------------- semantic-change hash ---

def test_changed_camera_extrinsic_changes_hash():
    m1 = make_rollout_manifest(camera_config_hash="d" * 64)
    m2 = make_rollout_manifest(camera_config_hash="f" * 64)
    assert manifest_content_hash(m1) != manifest_content_hash(m2)


def test_changed_checkpoint_path_or_size_changes_checkpoint_hash(tmp_path):
    ckpt_a = tmp_path / "ckpt_a"
    ckpt_a.mkdir()
    (ckpt_a / "weights.bin").write_bytes(b"0" * 1024)

    ckpt_b = tmp_path / "ckpt_b"
    ckpt_b.mkdir()
    (ckpt_b / "weights.bin").write_bytes(b"0" * 2048)  # different size

    h_a = hash_checkpoint_path(ckpt_a)
    h_b = hash_checkpoint_path(ckpt_b)
    assert h_a != h_b

    # Same size+name at a different relative path (e.g. an extra subdir)
    # also changes the fingerprint, documenting the path-sensitivity half
    # of the "path+size+mtime, not bytes" contract.
    ckpt_c = tmp_path / "ckpt_c"
    (ckpt_c / "subdir").mkdir(parents=True)
    (ckpt_c / "subdir" / "weights.bin").write_bytes(b"0" * 1024)
    h_c = hash_checkpoint_path(ckpt_c)
    assert h_c != h_a


def test_changed_checkpoint_content_changes_hash_via_mtime(tmp_path):
    # Documents the module's stated limitation: content changes are only
    # visible through the size/mtime proxy, not through byte hashing.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    f = ckpt / "weights.bin"
    f.write_bytes(b"a" * 100)
    before = f.stat()
    h_before = hash_checkpoint_path(ckpt)

    import os
    f.write_bytes(b"b" * 100)
    # Exercise the documented metadata contract deterministically. Wall-clock
    # sleep/utime(None) can retain the same timestamp on the shared filesystem.
    os.utime(f, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    h_after = hash_checkpoint_path(ckpt)
    assert h_before != h_after


def test_checkpoint_hash_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_checkpoint_path(tmp_path / "does_not_exist")


def test_manifest_using_different_policy_checkpoint_hash_differs():
    ckpt_dir_1 = "e" * 64
    ckpt_dir_2 = "1" + "e" * 63
    m1 = make_rollout_manifest(policy_checkpoint_hash=ckpt_dir_1)
    m2 = make_rollout_manifest(policy_checkpoint_hash=ckpt_dir_2)
    assert manifest_content_hash(m1) != manifest_content_hash(m2)


# --------------------------------------------------------- write guard ----

def test_write_manifest_creates_file(tmp_path):
    m = make_rollout_manifest()
    target = manifest_io.write_manifest(tmp_path, m)
    assert target.exists()
    assert manifest_io.load_json(target) == m


def test_write_manifest_idempotent_rerun_succeeds(tmp_path):
    m = make_rollout_manifest()
    manifest_io.write_manifest(tmp_path, m)
    # Re-running with the exact same semantic manifest (e.g. RESUME=1) must
    # not raise, and must leave the directory in the same state.
    target = manifest_io.write_manifest(tmp_path, m)
    assert manifest_io.load_json(target) == m


def test_write_manifest_refuses_conflicting_overwrite(tmp_path):
    m1 = make_rollout_manifest(task_id="task_a")
    m2 = make_rollout_manifest(task_id="task_b")
    manifest_io.write_manifest(tmp_path, m1)
    with pytest.raises(manifest_io.ManifestConflictError):
        manifest_io.write_manifest(tmp_path, m2)
    # The original manifest must be untouched after the refused write.
    assert manifest_io.load_json(tmp_path / manifest_io.DEFAULT_MANIFEST_FILENAME) == m1


def test_write_manifest_strict_mode_refuses_even_identical_rewrite(tmp_path):
    m = make_rollout_manifest()
    manifest_io.write_manifest(tmp_path, m)
    with pytest.raises(manifest_io.ManifestConflictError):
        manifest_io.write_manifest(tmp_path, m, allow_overwrite_if_same=False)


def test_write_manifest_conflict_error_names_differing_fields(tmp_path):
    m1 = make_rollout_manifest(camera_config_hash="d" * 64)
    m2 = make_rollout_manifest(camera_config_hash="f" * 64)
    manifest_io.write_manifest(tmp_path, m1)
    with pytest.raises(manifest_io.ManifestConflictError, match="camera_config_hash"):
        manifest_io.write_manifest(tmp_path, m2)


# -------------------------------------------------- diff / frozen fields --

def test_diff_manifests_frozen_only_flags_camera_hash():
    m1 = make_rollout_manifest(camera_config_hash="d" * 64)
    m2 = make_rollout_manifest(camera_config_hash="f" * 64, success=False)
    diff = manifest_io.diff_manifests(m1, m2, frozen_only=True)
    assert "camera_config_hash" in diff
    # success is an outcome field, not a declared frozen field -> excluded
    assert "success" not in diff


def test_diff_manifests_all_fields_includes_outcome_fields():
    m1 = make_rollout_manifest(success=True)
    m2 = make_rollout_manifest(success=False)
    diff = manifest_io.diff_manifests(m1, m2, frozen_only=False)
    assert "success" in diff


def test_diff_manifests_identical_manifests_no_diff():
    m1 = make_rollout_manifest()
    m2 = make_rollout_manifest()
    assert manifest_io.diff_manifests(m1, m2, frozen_only=True) == {}
    assert manifest_io.diff_manifests(m1, m2, frozen_only=False) == {}


# ------------------------------------------------------------- schema -----

def test_rollout_manifest_requires_task_id():
    with pytest.raises(Exception):
        make_rollout_manifest(task_id=None)


def test_rollout_manifest_requires_rollout_seed():
    with pytest.raises(Exception):
        make_rollout_manifest(rollout_seed=None)


def test_scene_build_manifest_allows_null_task_fields():
    # A scene build isn't tied to one task/seed/policy; those must be
    # legally null rather than forced to a meaningless placeholder.
    m = make_build_manifest()
    assert m.task_id is None
    assert m.rollout_seed is None
    assert m.policy_checkpoint_hash is None


def test_extra_field_rejected():
    with pytest.raises(Exception):
        make_rollout_manifest(unexpected_field="surprise")


def test_manifest_from_dict_unknown_kind_raises():
    with pytest.raises(manifest_io.ManifestKindError):
        manifest_io.manifest_from_dict({"manifest_kind": "not_a_real_kind"})


# ------------------------------------------------------- semantic checks --

def test_find_semantic_issues_flags_cluster_path():
    d = {"checkpoint_path": "/group/worldcept/PhiRIE/checkpoints/openpi_cache/x"}
    issues = find_semantic_issues(d)
    assert any("cluster-specific path" in i for i in issues)


def test_find_semantic_issues_flags_nan():
    d = {"metric": float("nan")}
    issues = find_semantic_issues(d)
    assert any("non-finite" in i for i in issues)


def test_canonical_hash_rejects_nan():
    with pytest.raises(ValueError):
        canonical_hash({"x": float("nan")})


def test_example_manifest_validates():
    root = Path(__file__).resolve().parents[1]
    example = root / "configs" / "experiments" / "example_manifest.yaml"
    m = manifest_io.load_yaml(example)
    assert isinstance(m, RolloutManifest)
    # No semantic warnings expected: the shipped example is meant to be a
    # clean template, not a demonstration of a warning.
    assert find_semantic_issues(manifest_io.to_dict(m)) == []


# ------------------------------------------------------------------ CLI ---

def test_cli_validate_example_manifest(capsys):
    root = Path(__file__).resolve().parents[1]
    example = root / "configs" / "experiments" / "example_manifest.yaml"
    rc = manifest_io.main(["validate", str(example)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK:" in out
    assert "content_hash=" in out


def test_cli_validate_bad_file_returns_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"manifest_kind": "rollout"}))  # missing required fields
    rc = manifest_io.main(["validate", str(bad)])
    assert rc == 1


def test_cli_diff_reports_frozen_field_drift(tmp_path, capsys):
    m1 = make_rollout_manifest(camera_config_hash="d" * 64)
    m2 = make_rollout_manifest(camera_config_hash="f" * 64)
    p1, p2 = tmp_path / "m1.json", tmp_path / "m2.json"
    manifest_io.dump_json(m1, p1)
    manifest_io.dump_json(m2, p2)
    rc = manifest_io.main(["diff", str(p1), str(p2)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "camera_config_hash" in out
