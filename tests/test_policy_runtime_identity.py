from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robo.eval import paired_runner
from robo.eval import harness_runner
from robo.eval.harness_spec import load_harness_spec
from robo.eval import harness_validation
from robo.manifest.hash import canonical_hash, hash_checkpoint_path
from robo.policy.clients.pi05_client import Pi05PolicyClient
from robo.policy.registry import (
    CheckpointHashMismatchError,
    PolicyEntry,
    PolicyRegistry,
)
from robo.policy.runtime_identity import (
    IDENTITY_METADATA_KEY,
    PolicyRuntimeIdentityError,
    build_server_identity,
    stage_checkpoint,
    validate_server_identity,
    verify_clean_git_checkout,
)


def _git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked.txt").write_text("pinned\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _checkpoint(path: Path) -> tuple[Path, str]:
    path.mkdir()
    (path / "weights.bin").write_bytes(b"weights")
    (path / "metadata.json").write_text("{}\n")
    return path, hash_checkpoint_path(path)


def test_clean_git_checkout_is_exact_and_dirty_state_is_rejected(tmp_path):
    root = tmp_path / "openpi"
    commit = _git_repo(root)
    assert verify_clean_git_checkout(
        root, commit, owner="openpi") == {
            "root": str(root.resolve()), "commit": commit, "dirty": False}
    (root / "tracked.txt").write_text("dirty\n")
    with pytest.raises(PolicyRuntimeIdentityError, match="worktree is dirty"):
        verify_clean_git_checkout(root, commit, owner="openpi")


def test_checkpoint_cache_is_verified_not_sentinel_trusted(tmp_path):
    source, fingerprint = _checkpoint(tmp_path / "source")
    destination = stage_checkpoint(
        source_path=source, checkpoint_fingerprint=fingerprint,
        cache_root=tmp_path / "cache", policy_id="policy")
    assert hash_checkpoint_path(destination) == fingerprint
    assert not (destination / ".copy_complete").exists()

    (destination / "weights.bin").write_bytes(b"corrupt")
    repaired = stage_checkpoint(
        source_path=source, checkpoint_fingerprint=fingerprint,
        cache_root=tmp_path / "cache", policy_id="policy")
    assert repaired == destination
    assert hash_checkpoint_path(repaired) == fingerprint
    assert len(list(destination.parent.glob("checkpoint.invalid.*"))) == 1


def test_identity_binds_source_runtime_config_and_clean_openpi(tmp_path):
    openpi = tmp_path / "openpi"
    commit = _git_repo(openpi)
    source, fingerprint = _checkpoint(tmp_path / "source")
    runtime = stage_checkpoint(
        source_path=source, checkpoint_fingerprint=fingerprint,
        cache_root=tmp_path / "cache", policy_id="policy")
    identity = build_server_identity(
        policy_id="policy", checkpoint_path=source,
        checkpoint_fingerprint=fingerprint, training_config="train_config",
        openpi_root=openpi, openpi_commit=commit, port=8123,
        runtime_checkpoint_path_value=runtime)
    assert validate_server_identity(identity, expected=identity) == identity
    assert identity["policy"]["checkpoint"] == {
        "declared_source_path": str(source.resolve()),
        "runtime_path": str(runtime.resolve()),
        "fingerprint": fingerprint,
        "fingerprint_kind": "tree_path_size_mtime_sha256",
    }

    tampered = copy.deepcopy(identity)
    tampered["policy"]["training_config"] = "wrong"
    with pytest.raises(PolicyRuntimeIdentityError, match="identity_sha256"):
        validate_server_identity(tampered)


class _MetadataServer:
    def __init__(self, identity):
        self.identity = identity

    def get_server_metadata(self):
        return {IDENTITY_METADATA_KEY: self.identity}

    def infer(self, _request):  # pragma: no cover - identity check needs no inference
        return {"actions": np.zeros((15, 8))}


def test_pi05_client_authenticates_handshake_before_inference(tmp_path):
    openpi = tmp_path / "openpi"
    commit = _git_repo(openpi)
    checkpoint, fingerprint = _checkpoint(tmp_path / "checkpoint")
    identity = build_server_identity(
        policy_id="policy", checkpoint_path=checkpoint,
        checkpoint_fingerprint=fingerprint, training_config="config",
        openpi_root=openpi, openpi_commit=commit, port=8000)
    expected_policy = {
        "id": "policy", "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_fingerprint": fingerprint, "training_config": "config",
    }
    client = Pi05PolicyClient(
        connector=lambda: _MetadataServer(identity),
        expected_server_identity=identity,
        expected_policy_identity=expected_policy)
    assert client.verify_server_identity() == identity

    wrong = copy.deepcopy(identity)
    wrong["policy"]["training_config"] = "other"
    unsigned = copy.deepcopy(wrong)
    unsigned.pop("identity_sha256")
    wrong["identity_sha256"] = canonical_hash(unsigned)
    bad_client = Pi05PolicyClient(
        connector=lambda: _MetadataServer(wrong),
        expected_server_identity=identity,
        expected_policy_identity=expected_policy)
    with pytest.raises(PolicyRuntimeIdentityError, match="exact harness declaration"):
        bad_client.verify_server_identity()


def test_paired_runner_does_not_fallback_after_registry_failure(monkeypatch):
    class FailingRegistry:
        @classmethod
        def from_config_dir(cls):
            raise CheckpointHashMismatchError("mismatch")

    monkeypatch.setattr(paired_runner, "_HAVE_REGISTRY", True)
    monkeypatch.setattr(paired_runner, "PolicyRegistry", FailingRegistry)
    monkeypatch.setattr(
        paired_runner, "_frozen_policies",
        lambda: (_ for _ in ()).throw(AssertionError("fallback was used")))
    with pytest.raises(CheckpointHashMismatchError, match="mismatch"):
        paired_runner.get_policy("policy", np.zeros(7))


def test_harness_runtime_contract_records_declared_current_and_server_hash():
    fingerprint = "ab" * 32
    server_identity = {
        "schema_version": 1,
        "kind": "fixture",
        "policy": {"checkpoint": {"fingerprint": fingerprint}},
    }
    config = {
        "policy": "real_policy", "horizon_s": 32,
        "contract": {
            "policy": {
                "id": "real_policy", "kind": "real",
                "checkpoint_hash": fingerprint,
            },
            "runtime_dependencies": {"fixture": True},
        },
    }
    state = SimpleNamespace(
        ep=0, scene_id="scene", task_id="task", reset_state_id="reset", base_seed=0,
        reset_seed=1)
    contract = harness_runner._runtime_contract(
        config=config, state=state, controller_hash="controller",
        camera_hash="camera", action_convention="absolute_joint_position",
        action_dim=8, policy_hash=fingerprint,
        server_identity=server_identity, task_instruction="do task")
    assert contract["policy_checkpoint_hash"] == fingerprint
    assert contract["runtime_policy_checkpoint_fingerprint"] == fingerprint
    assert contract["policy"]["server_identity"] == server_identity

    with pytest.raises(ValueError, match="differs from current"):
        harness_runner._runtime_contract(
            config=config, state=state, controller_hash="controller",
            camera_hash="camera", action_convention="absolute_joint_position",
            action_dim=8, policy_hash="cd" * 32,
            server_identity=server_identity, task_instruction="do task")


def _real_entry(checkpoint: Path, fingerprint: str) -> PolicyEntry:
    return PolicyEntry(
        id="real_policy", status="verified", client_kind="pi05_server",
        checkpoint_path=str(checkpoint), checkpoint_hash=fingerprint,
        training_config="training_config",
        env_action_convention="absolute_joint_position", action_dim=8,
        gripper_range=(0.0, 1.0), gripper_open_value=0.0,
        gripper_closed_value=1.0, gripper_binarize_threshold=0.5,
        control_rate_hz=15, per_tick_joint_delta_clamp_rad=0.2,
    )


def test_e4_real_contract_requires_registry_and_clean_pinned_dependencies(
    tmp_path, monkeypatch,
):
    checkpoint, fingerprint = _checkpoint(tmp_path / "checkpoint")
    openpi = tmp_path / "openpi"
    openpi_commit = _git_repo(openpi)
    menagerie = tmp_path / "menagerie"
    menagerie_commit = _git_repo(menagerie)
    for relative in (
        "franka_emika_panda/panda_nohand.xml",
        "robotiq_2f85/2f85.xml",
    ):
        path = menagerie / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mujoco/>\n")
    subprocess.run(["git", "-C", str(menagerie), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(menagerie), "commit", "-qm", "rig assets"], check=True)
    menagerie_commit = subprocess.check_output(
        ["git", "-C", str(menagerie), "rev-parse", "HEAD"], text=True).strip()
    registry = PolicyRegistry.from_entries([_real_entry(checkpoint, fingerprint)])
    monkeypatch.setattr(
        harness_validation.PolicyRegistry, "from_config_dir",
        classmethod(lambda cls: registry))
    config = {
        "policy": "real_policy", "port": 8123, "paper_mode": False,
        "contract": {
            "policy": {
                "id": "real_policy", "kind": "real",
                "checkpoint_path": str(checkpoint),
                "checkpoint_hash": fingerprint,
                "checkpoint_hash_kind": "tree_path_size_mtime_sha256",
                "training_config": "training_config",
            },
            "runtime_dependencies": {
                "openpi": {"root": str(openpi), "commit": openpi_commit},
                "mujoco_menagerie": {
                    "root": str(menagerie), "commit": menagerie_commit},
                "checkpoint_cache_root": "/scratch/runyi_yang/e4-policy-test-cache",
            },
            "robot": {"id": "robot"}, "cameras": {"id": "cameras"},
            "action_convention": "absolute_joint_position",
            "controller": {"id": "controller"}, "control_rate_hz": 15,
            "horizon_s": 32, "task_instruction": "task",
            "rubric": {"id": "rubric"}, "reset_ids": ["r0"],
        },
        "treatments": [
            {"id": "a0", "scene": "fixed_single_path"},
            {"id": "a4", "scene": "agentic"},
        ],
        "comparisons": [{
            "id": "construction", "axis": "scene", "treatments": ["a0", "a4"],
        }],
    }
    assert load_harness_spec(config).raw == config
    (openpi / "untracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="openpi worktree is dirty"):
        load_harness_spec(config)
