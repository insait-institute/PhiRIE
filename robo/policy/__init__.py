"""robo.policy: policy/control/checkpoint registry (Task 08,
plan/08_POLICY_CONTROL_CHECKPOINT_MATRIX.md).

Runs a non-degenerate matrix of policies whose checkpoint bytes correspond
to real-world results, with identical preprocessing/control across every
simulator variant. Three pieces:

  - `robo.policy.control_contract`: the frozen, environment-facing control
    contract (`ControlContract`, `FROZEN_CONTROL_CONTRACT`) every client
    normalizes to, plus the `PolicyClient` interface (explicit `warmup()`,
    schema-validated `__call__`) and `run_trace()` for deterministic
    observation/action replay.
  - `robo.policy.registry`: loads `configs/policies/*.yaml` into typed
    `PolicyEntry` records (status: verified / exploratory / unavailable),
    verifies pinned checkpoint hashes (`robo.manifest.hash`), validates
    each entry against the frozen contract, and constructs runnable
    clients via `PolicyRegistry.make_client`.
  - `robo.policy.clients`: adapters (`Pi05PolicyClient`,
    `ScriptedPolicyClient`) porting the real client/server contract from
    `robo/eval/pi05_eval.py` / `run/pi05_serve.sh` onto the shared
    `PolicyClient` interface, without modifying those files.

    from robo.policy.registry import PolicyRegistry
    from robo.policy.control_contract import FROZEN_CONTROL_CONTRACT, run_trace

    registry = PolicyRegistry.from_config_dir()
    client = registry.make_client("scripted_sinusoid", home=home_pose)
"""
from robo.policy.control_contract import (
    ControlContract,
    FROZEN_CONTROL_CONTRACT,
    PolicyClient,
    run_trace,
)
from robo.policy.registry import PolicyEntry, PolicyRegistry

__all__ = [
    "ControlContract",
    "FROZEN_CONTROL_CONTRACT",
    "PolicyClient",
    "run_trace",
    "PolicyEntry",
    "PolicyRegistry",
]
