from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robo.eval import freeze


def _write_tree(root: Path, *, ordered_value: dict | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "paper").mkdir()
    (root / "metadata.json").write_text('{"source": "fixture"}\n')
    config_dir = root / "configs" / "experiments" / "icra2027"
    config_dir.mkdir(parents=True)
    references = {
        "construction_config": "construction_regimes.yaml",
        "fidelity_manifest": "fidelity_manifest.json",
        "harness_config": "harness.yaml",
        "audit_config": "audit.yaml",
        "harmony_manifest": "harmony_visual_manifest.json",
        "real_world_config": "real_world.yaml",
    }
    value = ordered_value or {"alpha": 1, "beta": {"x": 2, "y": 3}}
    for filename in references.values():
        path = config_dir / filename
        if path.suffix == ".json":
            path.write_text(json.dumps(value) + "\n")
        else:
            path.write_text(yaml.safe_dump(value, sort_keys=False))
    root_config = {
        "schema_version": 1,
        "mode": "smoke",
        "freeze_id": "fixture-v1",
        "code_commit": "auto",
        "git_dirty": False,
        "created_utc": "auto",
        "paper_repository": "paper",
        "paper_commit_before_update": "auto",
        "input_roots": [{
            "id": "fixture-metadata", "path": "metadata.json",
            "kind": "metadata", "required": True,
        }],
        "checkpoint_roots": [],
        "hardware": {"accelerator": "cpu"},
        **{field: f"configs/experiments/icra2027/{filename}"
           for field, filename in references.items()},
    }
    root_path = config_dir / "freeze.yaml"
    root_path.write_text(yaml.safe_dump(root_config, sort_keys=False))
    return root_path


@pytest.fixture
def fake_git(monkeypatch):
    snapshot = {"commit": "abc1234", "dirty": False, "branch": "fixture"}
    monkeypatch.setattr(freeze, "git_snapshot", lambda _root: dict(snapshot))
    return snapshot


def test_dirty_worktree_rejected_unless_explicit_smoke_override(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    config = _write_tree(root)
    monkeypatch.setattr(
        freeze, "git_snapshot",
        lambda _root: {"commit": "abc1234", "dirty": True, "branch": "fixture"},
    )
    with pytest.raises(freeze.FreezeError, match="working tree is dirty"):
        freeze.create_freeze(config, root / "out", repo_root=root, dry_run=True)
    manifest = freeze.create_freeze(
        config, root / "out", repo_root=root, dry_run=True,
        allow_dirty_for_smoke=True,
    )
    assert manifest["code"]["dirty_override_for_smoke"] is True


def test_existing_output_is_never_overwritten(tmp_path, fake_git):
    root = tmp_path / "repo"
    config = _write_tree(root)
    output = root / "contract"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not replace\n")
    with pytest.raises(freeze.FreezeError, match="refusing to overwrite"):
        freeze.create_freeze(config, output, repo_root=root)
    assert marker.read_text() == "do not replace\n"


def test_equivalent_yaml_key_order_has_same_semantic_hash(tmp_path, fake_git):
    root = tmp_path / "repo"
    config = _write_tree(root, ordered_value={"alpha": 1, "beta": 2})
    first = freeze.create_freeze(config, root / "one", repo_root=root, dry_run=True)
    target = root / "configs" / "experiments" / "icra2027" / "harness.yaml"
    target.write_text("beta: 2\nalpha: 1\n")
    second = freeze.create_freeze(config, root / "two", repo_root=root, dry_run=True)
    first_hash = next(row["sha256"] for row in first["configs"]
                      if row["field"] == "harness_config")
    second_hash = next(row["sha256"] for row in second["configs"]
                       if row["field"] == "harness_config")
    assert first_hash == second_hash


def test_freeze_writes_seven_configs_hashes_and_resource_inventory(tmp_path, fake_git):
    root = tmp_path / "repo"
    config = _write_tree(root)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"fixture-weights")
    root_value = yaml.safe_load(config.read_text())
    root_value["checkpoint_roots"] = [{
        "id": "fixture-checkpoint", "path": "checkpoint",
        "kind": "checkpoint", "required": True,
    }]
    config.write_text(yaml.safe_dump(root_value, sort_keys=False))
    preflight = root / "preflight-source.log"
    preflight.write_text("preflight passed\n")
    output = root / "contract"
    manifest = freeze.create_freeze(
        config, output, repo_root=root, preflight_log=preflight,
        now_utc="2026-09-03T00:00:00Z",
    )
    assert len(list((output / "resolved_configs").iterdir())) == 7
    assert (output / "freeze_manifest.json").is_file()
    assert (output / "freeze_report.txt").is_file()
    assert (output / "hashes.csv").is_file()
    assert (output / "preflight.log").read_text() == "preflight passed\n"
    inventory = json.loads((output / "resource_inventory.json").read_text())
    resources = inventory["resources"]
    metadata = next(row for row in resources if row["id"] == "fixture-metadata")
    assert metadata["exists"] is True
    assert metadata["type"] == "file"
    assert metadata["size_bytes"] > 0
    assert len(metadata["sha256"]) == 64
    assert len([row for row in resources if row["category"] == "config"]) == 7
    checkpoint_row = next(row for row in resources
                          if row["id"] == "fixture-checkpoint")
    assert checkpoint_row["type"] == "directory"
    assert checkpoint_row["size_bytes"] == len(b"fixture-weights")
    assert checkpoint_row["hash_method"] == "tree_path_size_mtime_sha256"
    assert len(checkpoint_row["sha256"]) == 64
    assert manifest["contract_sha256"]


def test_resource_under_tmp_is_rejected_before_fingerprinting(tmp_path, fake_git,
                                                               monkeypatch):
    root = tmp_path / "repo"
    config = _write_tree(root)
    value = yaml.safe_load(config.read_text())
    value["input_roots"] = [{
        "id": "forbidden", "path": "/tmp/simany-forbidden",
        "kind": "large_directory", "required": False,
    }]
    config.write_text(yaml.safe_dump(value, sort_keys=False))
    called = False

    def _unexpected_fingerprint(_path):
        nonlocal called
        called = True
        raise AssertionError("fingerprint must not be called")

    monkeypatch.setattr(freeze, "hash_checkpoint_path", _unexpected_fingerprint)
    with pytest.raises(freeze.FreezeError, match="forbidden /tmp"):
        freeze.create_freeze(config, root / "contract", repo_root=root, dry_run=True)
    assert called is False


def test_injected_exception_leaves_no_published_or_staging_directory(tmp_path, fake_git):
    root = tmp_path / "repo"
    config = _write_tree(root)
    output = root / "contract"
    with pytest.raises(RuntimeError, match="injected freeze failure"):
        freeze.create_freeze(
            config, output, repo_root=root,
            _inject_failure_after="inventory",
        )
    assert not output.exists()
    assert list(root.glob(".contract.staging-*")) == []


def test_auto_ids_skip_existing_and_failed_attempts(tmp_path, fake_git, monkeypatch):
    root = tmp_path / "repo"
    config = _write_tree(root)
    monkeypatch.setattr(freeze.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(freeze, "_environment_snapshot", lambda: {})
    outputs = root / "outputs"
    (outputs / "20260904-aaaaaaa-v1").mkdir(parents=True)
    kwargs = dict(repo_root=root, auto_freeze_id=True, now_utc="2026-09-04T00:00:00Z")
    with pytest.raises(RuntimeError, match="injected freeze failure"):
        freeze.create_freeze(config, outputs, _inject_failure_after="inventory", **kwargs)
    assert not (outputs / "20260904-aaaaaaa-v2" / "contract").exists()
    manifest = freeze.create_freeze(config, outputs, **kwargs)
    assert manifest["freeze_id"] == "20260904-aaaaaaa-v3"
    assert manifest["paper_ready"] is False
    assert (outputs / manifest["freeze_id"] / "contract" / "freeze_manifest.json").exists()


def test_auto_dry_run_reserves_nothing_and_uses_utc(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze.subprocess, "check_output", lambda *a, **kw: "b" * 40)
    outputs = tmp_path / "outputs"
    value = freeze.reserve_freeze_id(tmp_path, outputs,
                                    now_utc="2026-09-05T01:00:00+03:00", dry_run=True)
    assert value == "20260904-bbbbbbb-v1"
    assert not outputs.exists()


def test_auto_ids_are_unique_under_concurrent_allocation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(freeze.subprocess, "check_output", lambda *a, **kw: "c" * 40)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: freeze.reserve_freeze_id(
            tmp_path, tmp_path / "outputs", now_utc="2026-09-04T00:00:00Z"), range(8)))
    assert set(values) == {f"20260904-ccccccc-v{i}" for i in range(1, 9)}


def test_legacy_paper_id_rejected_but_smoke_stays_compatible(tmp_path, fake_git):
    root = tmp_path / "repo"
    config = _write_tree(root)
    value = yaml.safe_load(config.read_text())
    value["mode"] = "paper"
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(freeze.FreezeError, match="paper freeze_id must be"):
        freeze.create_freeze(config, root / "out", repo_root=root, dry_run=True)


def test_auto_cli_requires_output_root_pair():
    assert freeze.main(["--config", "unused", "--out-root", "unused"]) == 2
    assert freeze.main(["--config", "unused", "--out", "unused", "--auto-freeze-id"]) == 2
