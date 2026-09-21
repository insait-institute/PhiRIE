"""robo.policy.clients.scripted_client: deterministic scripted policy,
ported from robo/eval/pi05_eval.py::ScriptedPolicy onto the shared
robo.policy.control_contract.PolicyClient interface.

frozen_fields.yaml: `scripted_sinusoid`, `checkpoint_path: null`,
`action_output: scripted_smoke_test_only` -- this is the env/scorer smoke
test, not a claim about any real policy's behavior, so it has no checkpoint
and no model to warm up.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from robo.policy.control_contract import ControlContract, PolicyClient

# Fixed per-joint sway weights, ported verbatim from
# robo/eval/pi05_eval.py::ScriptedPolicy.__call__ -- changing these changes
# the scripted smoke test's behavior, so they live here as a named constant
# rather than being re-derived at each call site.
_SWAY_WEIGHTS = np.array([1.0, 0.5, 0.0, 0.4, 0.0, -0.4, 0.0])
_SWAY_PERIOD = 22.0
_GRIPPER_TOGGLE_TICKS = 45


class ScriptedPolicyClient(PolicyClient):
    """Sinusoidal joint sway + periodic gripper open/close, deterministic
    given `home` and the tick counter alone (no observation dependence,
    matching the original `ScriptedPolicy`: it never reads `obs`).

    `warmup()` is a documented no-op -- there's no model to jit -- but it
    still exists (rather than being omitted) so a caller that always warms
    up every registry client before timing (per docs/ROBOT.md) doesn't need
    an isinstance check to skip it for this one client kind.
    """

    def __init__(self, home, *, strict_warmup: bool = False,
                 control_contract: "ControlContract | None" = None):
        super().__init__(strict_warmup=strict_warmup, control_contract=control_contract)
        self.home = np.asarray(home, dtype=float)
        self.t = 0

    @classmethod
    def from_entry(cls, entry, *, home, **kwargs) -> "ScriptedPolicyClient":
        """Build from a `robo.policy.registry.PolicyEntry`. `home` (the
        robot's reset joint pose, e.g. `robo.envs.pi05_env.DroidSimEnv.
        info["home"]`) has no config-file representation -- it's an
        environment fact, not a policy fact -- so it's a required keyword
        here rather than an entry field."""
        return cls(home=home, **kwargs)

    def _warmup_impl(self, obs: Mapping, prompt: str) -> None:
        pass  # no model to jit -- see class docstring

    def reset(self) -> None:
        self.t = 0

    def _act(self, obs: Mapping, prompt: str) -> np.ndarray:
        self.t += 1
        a = np.zeros(self.control_contract.action_dim)
        a[:7] = self.home + 0.25 * np.sin(self.t / _SWAY_PERIOD) * _SWAY_WEIGHTS
        a[7] = (self.control_contract.gripper_closed_value
                if (self.t // _GRIPPER_TOGGLE_TICKS) % 2
                else self.control_contract.gripper_open_value)
        return a


__all__ = ["ScriptedPolicyClient"]
