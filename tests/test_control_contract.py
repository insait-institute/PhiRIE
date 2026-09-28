"""Tests for robo.policy (Task 08,
plan/08_POLICY_CONTROL_CHECKPOINT_MATRIX.md).

Run with:
    .venv/bin/python -m pytest -q tests/test_control_contract.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from robo.manifest.hash import hash_checkpoint_path
from robo.policy.clients.pi05_client import Pi05PolicyClient
from robo.policy.clients.scripted_client import ScriptedPolicyClient
from robo.policy.control_contract import (
    ControlContractMismatchError,
    EnvActionShapeError,
    FROZEN_CONTROL_CONTRACT,
    PolicyNotWarmedUpError,
    assert_matches_frozen,
    run_trace,
    validate_against_frozen,
    validate_env_action,
)
from robo.policy.registry import (
    CheckpointHashMismatchError,
    PolicyEntry,
    PolicyRegistry,
    PolicyUnavailableError,
    UnknownPolicyError,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "policies"


def make_entry(**overrides) -> PolicyEntry:
    """A minimal, frozen-contract-compliant PolicyEntry for tests that want
    to mutate exactly one field (mirrors tests/test_manifest_roundtrip.py's
    make_rollout_manifest helper pattern)."""
    fields = dict(
        id="test_policy",
        status="verified",
        client_kind="scripted",
        env_action_convention="absolute_joint_position",
        action_dim=8,
        gripper_range=(0.0, 1.0),
        gripper_open_value=0.0,
        gripper_closed_value=1.0,
        gripper_binarize_threshold=0.5,
        control_rate_hz=15,
        per_tick_joint_delta_clamp_rad=0.2,
    )
    fields.update(overrides)
    return PolicyEntry(**fields)


def make_obs(t: int = 0) -> dict:
    """A golden observation matching robo/envs/pi05_env.py::DroidSimEnv's
    observation dict shape exactly."""
    return {
        "observation/exterior_image_1_left": np.full(
            (360, 640, 3), fill_value=t % 256, dtype=np.uint8),
        "observation/wrist_image_left": np.full(
            (360, 640, 3), fill_value=(t + 1) % 256, dtype=np.uint8),
        "observation/joint_position": np.zeros(7) + 0.01 * t,
        "observation/gripper_position": np.array([0.0]),
    }


GOLDEN_TRACE = [make_obs(t) for t in range(20)]


class FakeServer:
    """Stands in for openpi_client.websocket_client_policy.
    WebsocketClientPolicy: same `.infer(request) -> {"actions": ndarray}`
    surface, no network. Always returns a fixed (chunk_size, 8) chunk of
    ABSOLUTE joint targets + gripper, mirroring the real post-
    AbsoluteActions-transform response shape documented in docs/ROBOT.md."""

    def __init__(self, row=None, chunk_size=15):
        self.row = np.array(row if row is not None else
                             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
        self.chunk_size = chunk_size
        self.n_infer_calls = 0

    def infer(self, request):
        self.n_infer_calls += 1
        assert request["observation/exterior_image_1_left"].shape == (224, 224, 3)
        assert request["observation/wrist_image_left"].shape == (224, 224, 3)
        return {"actions": np.tile(self.row, (self.chunk_size, 1))}


# ------------------------------------------------------- golden replay -----
# Task 08 test: "Golden observation produces a stable action trace within
# numerical tolerance" / step 5 "deterministic observation/action trace
# replay".

def test_scripted_client_golden_trace_is_stable_across_repeated_runs():
    home = np.array([0.0, -0.6283185307, 0.0, -2.5132741229, 0.0, 1.8849555922, 0.0])
    trace_a = run_trace(ScriptedPolicyClient(home=home), GOLDEN_TRACE, "warmup")
    trace_b = run_trace(ScriptedPolicyClient(home=home), GOLDEN_TRACE, "warmup")
    assert len(trace_a) == len(GOLDEN_TRACE)
    for a, b in zip(trace_a, trace_b):
        np.testing.assert_array_equal(a, b)


def test_scripted_client_trace_matches_hand_derived_formula():
    # Cross-check against the exact formula in
    # robo/policy/clients/scripted_client.py (ported from
    # robo/eval/pi05_eval.py::ScriptedPolicy) rather than only comparing two
    # runs to each other, so a change to the formula itself is caught too.
    home = np.zeros(7)
    trace = run_trace(ScriptedPolicyClient(home=home), GOLDEN_TRACE, "warmup")
    weights = np.array([1.0, 0.5, 0.0, 0.4, 0.0, -0.4, 0.0])
    for t, action in enumerate(trace, start=1):
        expected_arm = home + 0.25 * np.sin(t / 22.0) * weights
        np.testing.assert_allclose(action[:7], expected_arm)
        expected_gripper = 1.0 if (t // 45) % 2 else 0.0
        assert action[7] == expected_gripper


def test_pi05_client_mock_trace_is_deterministic_and_matches_fixed_stub():
    row = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]

    def build_client():
        server = FakeServer(row=row)
        return Pi05PolicyClient(connector=lambda: server, strict_warmup=True), server

    client_a, server_a = build_client()
    trace_a = run_trace(client_a, GOLDEN_TRACE, "put the mug on the tray",
                         warmup_obs=make_obs(0))
    client_b, server_b = build_client()
    trace_b = run_trace(client_b, GOLDEN_TRACE, "put the mug on the tray",
                         warmup_obs=make_obs(0))

    assert len(trace_a) == len(GOLDEN_TRACE)
    for a, b in zip(trace_a, trace_b):
        np.testing.assert_array_equal(a, b)
    for a in trace_a:
        np.testing.assert_allclose(a, row)

    # 20 ticks at chunk_size=15 -> requeries at tick 0 and tick 15, plus the
    # one warmup call = 3 real infer() calls, not 20 (Task 08 step 1:
    # "chunking, 15-step chunks"; docs/ROBOT.md: "executes the full 15-step
    # chunk ... then requeries").
    assert server_a.n_infer_calls == 3
    assert server_b.n_infer_calls == 3


def test_pi05_client_requires_warmup_before_call_when_strict():
    client = Pi05PolicyClient(connector=lambda: FakeServer(), strict_warmup=True)
    with pytest.raises(PolicyNotWarmedUpError):
        client(make_obs(0), "go")


# ------------------------------------------------------- hash mismatch -----
# Task 08 test: "Hash mismatch prevents a run from entering the main
# matrix" -- and the ownership brief's "registry correctly rejects/flags a
# policy config with checkpoint hash mismatch from what's declared."

def test_registry_detects_checkpoint_hash_mismatch(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"0" * 1024)
    real_hash = hash_checkpoint_path(ckpt)
    wrong_hash = "f" * 64
    assert real_hash != wrong_hash

    entry = make_entry(checkpoint_path=str(ckpt), checkpoint_hash=wrong_hash)
    registry = PolicyRegistry.from_entries([entry])

    with pytest.raises(CheckpointHashMismatchError):
        registry.verify_checkpoint_hash("test_policy")
    with pytest.raises(CheckpointHashMismatchError):
        registry.make_client("test_policy", home=np.zeros(7))


def test_registry_accepts_matching_checkpoint_hash(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.bin").write_bytes(b"0" * 1024)
    real_hash = hash_checkpoint_path(ckpt)

    entry = make_entry(checkpoint_path=str(ckpt), checkpoint_hash=real_hash)
    registry = PolicyRegistry.from_entries([entry])
    registry.verify_checkpoint_hash("test_policy")  # must not raise
    client = registry.make_client("test_policy", home=np.zeros(7))
    assert isinstance(client, ScriptedPolicyClient)


def test_registry_missing_checkpoint_on_disk_is_not_a_hash_mismatch(tmp_path):
    # An absent checkpoint is PolicyUnavailableError territory (checked
    # earlier in make_client), not a hash-mismatch error -- verify_checkpoint_hash
    # itself must stay a no-op when the path just isn't there to compare
    # against (robo.policy.registry docstring: "nothing to compare against").
    entry = make_entry(checkpoint_path=str(tmp_path / "does_not_exist"),
                        checkpoint_hash="a" * 64)
    registry = PolicyRegistry.from_entries([entry])
    registry.verify_checkpoint_hash("test_policy")  # must not raise


# --------------------------------------------- swapped action convention ---
# Task 08 test: "Intentionally swapped absolute/delta convention is
# detected."

def test_swapped_env_action_convention_is_detected():
    swapped = make_entry(env_action_convention="joint_delta")
    issues = validate_against_frozen(swapped)
    assert any("env_action_convention" in issue for issue in issues)
    with pytest.raises(ControlContractMismatchError):
        assert_matches_frozen(swapped)


def test_correct_env_action_convention_passes():
    correct = make_entry(env_action_convention="absolute_joint_position")
    assert validate_against_frozen(correct) == []
    assert_matches_frozen(correct)  # must not raise


def test_frozen_contract_env_action_convention_is_absolute_joint_position():
    # Regression guard tying this module directly to
    # configs/experiments/frozen_fields.yaml's control.action_convention,
    # and to the documented model-predicts-delta /
    # server-converts-to-absolute distinction (docs/ROBOT.md) this module
    # exists to encode correctly.
    assert FROZEN_CONTROL_CONTRACT.env_action_convention == "absolute_joint_position"


# ------------------------------------------------- gripper convention -----
# Task 08 test: "Reversed/out-of-range gripper convention is detected
# similarly."

def test_reversed_gripper_direction_is_detected():
    reversed_gripper = make_entry(gripper_open_value=1.0, gripper_closed_value=0.0)
    issues = validate_against_frozen(reversed_gripper)
    assert any("gripper_direction" in issue for issue in issues)
    with pytest.raises(ControlContractMismatchError):
        assert_matches_frozen(reversed_gripper)


def test_out_of_range_gripper_range_is_detected():
    out_of_range = make_entry(gripper_range=(-1.0, 1.0))
    issues = validate_against_frozen(out_of_range)
    assert any("gripper_range" in issue for issue in issues)


def test_wrong_gripper_binarize_threshold_is_detected():
    wrong_threshold = make_entry(gripper_binarize_threshold=0.9)
    issues = validate_against_frozen(wrong_threshold)
    assert any("gripper_binarize_threshold" in issue for issue in issues)


def test_env_action_gripper_out_of_range_raises_at_runtime():
    # Runtime half of the same guarantee: even if a config-level check were
    # skipped, a single out-of-range gripper command from a client is
    # caught by validate_env_action (wired into every PolicyClient.__call__).
    bad_action = np.array([0, 0, 0, 0, 0, 0, 0, 5.0])
    with pytest.raises(EnvActionShapeError):
        validate_env_action(bad_action)


def test_env_action_wrong_dim_raises_at_runtime():
    with pytest.raises(EnvActionShapeError):
        validate_env_action(np.zeros(7))  # missing the gripper channel


def test_env_action_small_gripper_excursion_is_clamped_not_rejected():
    # Found 2026-08-31: ordinary model output noise (e.g. -0.003, 1.002) is
    # NOT a client bug -- the env binarizes the gripper channel at 0.5
    # downstream regardless, so a strict reject here killed ~25 otherwise-
    # normal real-policy episodes. Small excursions clamp; only something
    # far outside the range still raises (test above, 5.0).
    for bad, expected in [(-0.0025940738636255267, 0.0), (1.003, 1.0)]:
        action = np.array([0, 0, 0, 0, 0, 0, 0, bad])
        result = validate_env_action(action)
        assert result[-1] == expected


def test_env_action_accepts_read_only_input_array():
    # Found 2026-08-31: the clamp fix above mutates the gripper channel
    # in place. np.asarray (unlike np.array) returns a VIEW when the input
    # is already a float64 array -- if that input happens to be read-only
    # (e.g. a slice of a chunk decoded from a websocket response, as in
    # real robo.policy.clients.pi05_client usage), an in-place write raised
    # "ValueError: assignment destination is read-only" on every single
    # real-policy episode until this was caught.
    action = np.array([0, 0, 0, 0, 0, 0, 0, -0.002])
    action.flags.writeable = False
    result = validate_env_action(action)
    assert result[-1] == 0.0


# ------------------------------------------------- other frozen fields -----

def test_wrong_control_rate_is_detected():
    wrong_rate = make_entry(control_rate_hz=30)
    issues = validate_against_frozen(wrong_rate)
    assert any("control_rate_hz" in issue for issue in issues)


def test_wrong_joint_delta_clamp_is_detected():
    wrong_clamp = make_entry(per_tick_joint_delta_clamp_rad=1.0)
    issues = validate_against_frozen(wrong_clamp)
    assert any("per_tick_joint_delta_clamp_rad" in issue for issue in issues)


def test_frozen_contract_matches_frozen_fields_yaml_numbers():
    # configs/experiments/frozen_fields.yaml's literal values, so a silent
    # drift between that file and this module's fallback defaults is
    # caught immediately.
    assert FROZEN_CONTROL_CONTRACT.rate_hz == 15
    assert FROZEN_CONTROL_CONTRACT.action_dim == 8
    assert FROZEN_CONTROL_CONTRACT.gripper_binarize_threshold == 0.5
    assert FROZEN_CONTROL_CONTRACT.per_tick_joint_delta_clamp_rad == 0.2
    assert FROZEN_CONTROL_CONTRACT.horizon_seconds == 32
    # Not in frozen_fields.yaml -- pinned from docs/ROBOT.md instead (see
    # control_contract.py's load_frozen_control_contract docstring).
    assert FROZEN_CONTROL_CONTRACT.chunk_size == 15
    assert FROZEN_CONTROL_CONTRACT.temporal_aggregation == "requery_after_full_chunk"
    assert FROZEN_CONTROL_CONTRACT.image_resize_hw == (224, 224)


# --------------------------------------------------- unavailable policies --
# Task 08 test: unavailable-status policies cannot be instantiated into a
# runnable client -- the factory raises a clear error naming the missing
# checkpoint, not a generic crash.

def test_unavailable_policies_raise_named_error_not_generic_crash():
    registry = PolicyRegistry.from_config_dir(CONFIG_DIR)
    unavailable_ids = [pid for pid in registry.list_ids()
                        if registry.get(pid).status == "unavailable"]
    # plan/08's aspirational matrix (pi0-FAST, pi0, pi0-100k,
    # PaliGemma-binning) is at least three entries.
    assert len(unavailable_ids) >= 3

    for pid in unavailable_ids:
        entry = registry.get(pid)
        assert entry.client_kind == "unavailable"
        with pytest.raises(PolicyUnavailableError) as excinfo:
            registry.make_client(pid)
        message = str(excinfo.value)
        assert pid in message
        # The error must name the missing checkpoint, not just say "no".
        assert entry.checkpoint_uri in message


def test_unknown_policy_id_raises_unknown_policy_error():
    registry = PolicyRegistry.from_entries([make_entry()])
    with pytest.raises(UnknownPolicyError):
        registry.get("nonexistent_policy_id")


# ---------------------------------------------- real shipped configs ------
# Integration checks against the actual configs/policies/*.yaml this task
# ships, so a future edit to those files that breaks the contract fails a
# test immediately rather than only showing up in a real eval run.

def test_shipped_registry_has_exactly_the_frozen_fields_policies_plus_stubs():
    registry = PolicyRegistry.from_config_dir(CONFIG_DIR)
    ids = set(registry.list_ids())
    for required in ("pi05_droid_jointpos",
                      "droid_pi05_jointpos_with_web_and_sim",
                      "scripted_sinusoid"):
        assert required in ids, f"{required} (named in frozen_fields.yaml) missing from registry"


def test_shipped_registry_entries_all_pass_control_contract_validation():
    registry = PolicyRegistry.from_config_dir(CONFIG_DIR)
    for pid in registry.list_ids():
        issues = registry.validate_control_contract(pid)
        assert issues == [], f"{pid}: {issues}"


def test_shipped_pi05_policies_are_verified_or_exploratory_not_unavailable():
    registry = PolicyRegistry.from_config_dir(CONFIG_DIR)
    for pid in ("pi05_droid_jointpos", "droid_pi05_jointpos_with_web_and_sim"):
        assert registry.get(pid).status in ("verified", "exploratory")
        assert registry.get(pid).client_kind == "pi05_server"


@pytest.mark.skipif(
    not (Path(os.environ.get("OPENPI_DATA_HOME", "/nonexistent")) / "openpi-assets-simeval").exists(),
    reason="real checkpoint cache not present on this machine",
)
def test_shipped_real_checkpoint_hashes_match_disk_right_now():
    # Only meaningful on the cluster where the checkpoints actually live;
    # skipped elsewhere rather than failing on an irrelevant machine.
    registry = PolicyRegistry.from_config_dir(CONFIG_DIR)
    registry.verify_checkpoint_hash("pi05_droid_jointpos")
    registry.verify_checkpoint_hash("droid_pi05_jointpos_with_web_and_sim")
