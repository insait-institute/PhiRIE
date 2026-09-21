"""robo.policy.control_contract: the frozen, environment-facing control
contract every policy client under robo/policy/clients/ must normalize to,
plus the `PolicyClient` interface that enforces it.

Task 08 (plan/08_POLICY_CONTROL_CHECKPOINT_MATRIX.md). Source of truth for
every numeric value here: `configs/experiments/frozen_fields.yaml` (the
machine-readable frozen contract) and `docs/ROBOT.md` (the documented real
gotchas frozen_fields.yaml doesn't spell out: gripper direction, chunk
size, temporal aggregation, image resize, the JAX-jit warmup window).

**The one distinction this module exists to get right** (docs/ROBOT.md,
"robo/eval -- websocket policy client" section): pi0.5 predicts joint
*deltas* internally, but the openpi server-side `AbsoluteActions` transform
converts that prediction into a chunk of *absolute* joint-position targets
before the client (robo/eval/pi05_eval.py's `ServerPolicy`, and this
module's `Pi05PolicyClient`) ever sees it. So:

  - `ControlContract.env_action_convention` describes the CLIENT-FACING
    contract downstream of that transform -- always
    "absolute_joint_position" per frozen_fields.yaml, for every policy in
    the registry, pi0.5 included.
  - a policy's `model_action_convention` (a plain descriptive string on its
    `robo.policy.registry.PolicyEntry`, e.g. frozen_fields.yaml's
    "joint_delta_chunk_15x8") is informational only and is NOT validated
    against the frozen contract -- validating it would be validating the
    wrong half of the pipeline. Only `env_action_convention` is frozen.

Every value in `FROZEN_CONTROL_CONTRACT` must stay byte-identical across
every condition being compared, per docs/ICRA_RESEARCH_CONTRACT.md section
5; changing one requires a new frozen_fields.yaml version and a change-log
entry there, not a silent edit here.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FIELDS_PATH = ROOT / "configs" / "experiments" / "frozen_fields.yaml"

# 7 absolute joint targets + 1 gripper command, per frozen_fields.yaml
# `control.action_dim` -- the ONE environment-facing action schema every
# client in robo/policy/clients/ must produce, regardless of what the
# underlying model predicts internally.
ACTION_DIM = 8
JOINT_DIM = 7


# --------------------------------------------------------------- errors ---

class ControlContractError(Exception):
    """Base class for control-contract problems."""


class ControlContractMismatchError(ControlContractError):
    """A policy config disagrees with the frozen control contract
    (configs/experiments/frozen_fields.yaml) on a field that
    docs/ICRA_RESEARCH_CONTRACT.md section 5 requires byte-identical across
    every condition being compared."""


class PolicyNotWarmedUpError(ControlContractError):
    """`PolicyClient.__call__` invoked (with `strict_warmup=True`) before
    `warmup()`. Guards the real gotcha docs/ROBOT.md documents: first
    inference triggers a JAX jit that can take minutes, and the openpi
    websocket server's 20s keepalive kills the connection meanwhile if you
    don't eat that cost up front -- RoboLab's client and this repo's
    `ServerPolicy` (robo/eval/pi05_eval.py) both warm up once after
    connect/reconnect for exactly this reason."""


class EnvActionShapeError(ControlContractError):
    """A client returned something other than the frozen 8-dim
    (7 joint + 1 gripper) environment-facing action schema."""


# ---------------------------------------------------------- the contract --

@dataclass(frozen=True)
class ControlContract:
    """The frozen, environment-facing control contract. Not the model's
    internal action representation -- see module docstring. This is what
    `robo/envs/pi05_env.py::DroidSimEnv.apply_action` actually consumes.
    """

    rate_hz: int = 15
    physics_dt: float = 1.0 / 600.0
    substeps_per_tick: int = 40
    action_dim: int = ACTION_DIM
    env_action_convention: str = "absolute_joint_position"

    # Gripper: [0, 1], 0 = open, 1 = closed (docs/ROBOT.md /
    # robo/envs/pi05_env.py), binarized at 0.5 into ctrl 0/255.
    gripper_range: tuple[float, float] = (0.0, 1.0)
    gripper_open_value: float = 0.0
    gripper_closed_value: float = 1.0
    gripper_binarize_threshold: float = 0.5

    per_tick_joint_delta_clamp_rad: float = 0.2
    horizon_seconds: float = 32.0

    # Chunking + temporal aggregation (docs/ROBOT.md: "the client executes
    # the full 15-step chunk at 15 Hz, then requeries (RoboLab default)").
    # Not present in frozen_fields.yaml as of this writing; pinned here from
    # docs/ROBOT.md + robo/eval/pi05_eval.py's
    # `ServerPolicy(open_loop_horizon=15)` default.
    chunk_size: int = 15
    temporal_aggregation: str = "requery_after_full_chunk"

    # Image preprocessing: openpi_client.image_tools.resize_with_pad to
    # 224x224 (robo/eval/pi05_eval.py::ServerPolicy._request).
    image_resize_hw: tuple[int, int] = (224, 224)


def load_frozen_control_contract(
    path: "str | Path" = FROZEN_FIELDS_PATH,
) -> ControlContract:
    """Build the canonical `ControlContract` from
    configs/experiments/frozen_fields.yaml's `control` section, falling
    back to `ControlContract()`'s hardcoded defaults (with a loud warning,
    never a silent one) if the file is missing or malformed -- mirrors
    robo/eval/pi05_eval.py::_frozen_config_hashes()'s fallback so this
    module degrades the same way the eval script already does rather than
    crashing at import time if frozen_fields.yaml is mid-edit by another
    task's agent.
    """
    try:
        frozen = yaml.safe_load(Path(path).read_text())
        control = frozen["control"]
        return ControlContract(
            rate_hz=control["rate_hz"],
            physics_dt=control["physics_dt"],
            substeps_per_tick=control["substeps_per_tick"],
            action_dim=control["action_dim"],
            env_action_convention=control["action_convention"],
            gripper_binarize_threshold=control["gripper_binarize_threshold"],
            per_tick_joint_delta_clamp_rad=control["per_tick_joint_delta_clamp_rad"],
            horizon_seconds=control["horizon_seconds"],
        )
    except Exception as e:  # noqa: BLE001 -- see docstring: never crash at import
        print(f"[control_contract] WARNING: could not load {path} "
              f"({type(e).__name__}: {e}); falling back to this module's "
              f"hardcoded ControlContract() defaults instead of the frozen "
              f"contract file", flush=True)
        return ControlContract()


FROZEN_CONTROL_CONTRACT = load_frozen_control_contract()


# ------------------------------------------------------------ validation --

def _mismatch(name: str, declared: Any, expected: Any, extra: str = "") -> str:
    msg = f"{name} mismatch: declared {declared!r}, frozen contract requires {expected!r}"
    return f"{msg} ({extra})" if extra else msg


def validate_against_frozen(
    entry: Any, frozen: "ControlContract | None" = None,
) -> list[str]:
    """Check a policy config (any object exposing the attributes below --
    typically a `robo.policy.registry.PolicyEntry`, duck-typed rather than
    imported to avoid a registry<->control_contract import cycle) against
    the frozen control contract. Returns a list of human-readable mismatch
    strings (empty = fully compliant).

    Checked fields, each corresponding to a frozen_fields.yaml /
    docs/ROBOT.md value: `env_action_convention`, `action_dim`,
    `gripper_range`, `gripper_open_value`/`gripper_closed_value` (gripper
    *direction*, not just range), `gripper_binarize_threshold`,
    `control_rate_hz`, `per_tick_joint_delta_clamp_rad`.
    """
    frozen = frozen or FROZEN_CONTROL_CONTRACT
    issues: list[str] = []

    if entry.env_action_convention != frozen.env_action_convention:
        issues.append(_mismatch(
            "env_action_convention", entry.env_action_convention,
            frozen.env_action_convention,
            "docs/ROBOT.md: the model may predict joint DELTAS internally, "
            "but the server-side AbsoluteActions transform converts to a "
            "chunk of ABSOLUTE joint positions before the client ever sees "
            "it -- env_action_convention describes that client-facing "
            "contract, not the model's internal prediction target"))

    if int(entry.action_dim) != int(frozen.action_dim):
        issues.append(_mismatch("action_dim", entry.action_dim, frozen.action_dim))

    if tuple(entry.gripper_range) != tuple(frozen.gripper_range):
        issues.append(_mismatch("gripper_range", entry.gripper_range, frozen.gripper_range))

    if (float(entry.gripper_open_value) != float(frozen.gripper_open_value)
            or float(entry.gripper_closed_value) != float(frozen.gripper_closed_value)):
        issues.append(_mismatch(
            "gripper_direction",
            (entry.gripper_open_value, entry.gripper_closed_value),
            (frozen.gripper_open_value, frozen.gripper_closed_value),
            "docs/ROBOT.md: 0=open .. 1=closed -- this catches a reversed "
            "gripper convention even when gripper_range is otherwise correct"))

    if float(entry.gripper_binarize_threshold) != float(frozen.gripper_binarize_threshold):
        issues.append(_mismatch(
            "gripper_binarize_threshold", entry.gripper_binarize_threshold,
            frozen.gripper_binarize_threshold))

    if int(entry.control_rate_hz) != int(frozen.rate_hz):
        issues.append(_mismatch("control_rate_hz", entry.control_rate_hz, frozen.rate_hz))

    if float(entry.per_tick_joint_delta_clamp_rad) != float(frozen.per_tick_joint_delta_clamp_rad):
        issues.append(_mismatch(
            "per_tick_joint_delta_clamp_rad", entry.per_tick_joint_delta_clamp_rad,
            frozen.per_tick_joint_delta_clamp_rad))

    return issues


def assert_matches_frozen(entry: Any, frozen: "ControlContract | None" = None) -> None:
    """Raise `ControlContractMismatchError` if `validate_against_frozen`
    reports any issue; no-op otherwise."""
    issues = validate_against_frozen(entry, frozen)
    if issues:
        joined = "; ".join(issues)
        raise ControlContractMismatchError(
            f"policy config {getattr(entry, 'id', '<unknown>')!r} fails "
            f"control-contract validation against "
            f"configs/experiments/frozen_fields.yaml: {joined}")


def validate_env_action(
    action: "np.ndarray | Sequence[float]",
    contract: "ControlContract | None" = None,
) -> np.ndarray:
    """Validate (and return as a float ndarray) one environment-facing
    action: shape must be `(contract.action_dim,)`, no channel may be NaN/
    Inf, and the gripper channel (last element) is clamped into
    `contract.gripper_range`. This is the single point every
    `PolicyClient.__call__` routes through, so a client bug (wrong chunk
    slicing, wrong dim, NaN from a broken inference call) surfaces as a
    loud `EnvActionShapeError` before it reaches robo/envs/pi05_env.py
    instead of silently driving the robot with a malformed command.

    Small gripper excursions outside [0, 1] (e.g. -0.003, 1.002) are real,
    ordinary raw model output, not a bug: docs/ROBOT.md's own contract
    binarizes the gripper channel at a 0.5 threshold downstream, so
    anything within a wide margin of the valid range resolves to the same
    open/closed decision regardless. Found 2026-08-31: a strict `+/-1e-6`
    tolerance here rejected ~25 otherwise-normal episodes across all three
    mujoco_paired conditions with `EnvActionShapeError`, none of which was
    an actual client bug -- clamp instead of raising for anything inside a
    generous margin; only a channel far outside any plausible range (or
    NaN/Inf, which IS a real client bug) still raises.
    """
    contract = contract or FROZEN_CONTROL_CONTRACT
    # np.array (not np.asarray) forces a copy -- action may be a read-only
    # view (e.g. a slice of a chunk deserialized from the websocket
    # response), and the in-place gripper clamp below needs to write.
    a = np.array(action, dtype=float)
    if a.shape != (contract.action_dim,):
        raise EnvActionShapeError(
            f"expected a ({contract.action_dim},) env-facing action "
            f"({JOINT_DIM} joint + 1 gripper per frozen_fields.yaml), got "
            f"shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise EnvActionShapeError(f"action contains NaN/Inf: {a!r}")
    lo, hi = contract.gripper_range
    margin = 0.5 * (hi - lo)
    if not (lo - margin <= a[-1] <= hi + margin):
        raise EnvActionShapeError(
            f"gripper command {a[-1]!r} far outside frozen range "
            f"{contract.gripper_range} (margin {margin}) -- likely a real "
            f"client bug, not ordinary model noise")
    a[-1] = float(np.clip(a[-1], lo, hi))
    return a


# ---------------------------------------------------------- client base ---

class PolicyClient(abc.ABC):
    """Common interface every adapter under robo/policy/clients/ normalizes
    to. All clients speak the SAME environment-facing action schema
    (`ACTION_DIM` = 8: 7 absolute joint targets + 1 gripper in [0, 1]),
    whatever the underlying model predicts internally, per Task 08 step 2
    ("normalize all clients to one environment-facing action schema
    without changing model behavior").

    Subclasses implement `_act`, `_warmup_impl`, and `reset`; this base
    class enforces two things every subclass would otherwise have to
    reimplement:

    1. **Warmup is part of the interface, not an afterthought.**
       docs/ROBOT.md: first inference triggers a JAX jit that can take
       minutes and blows through the websocket's 20s keepalive -- every
       real closed-loop run must warm up once after connect/reconnect
       before any timed rollout. `warmup()` is concrete here (it sets
       `self._warmed_up`); `_warmup_impl` is what subclasses implement.
       Pass `strict_warmup=True` to make `__call__` raise
       `PolicyNotWarmedUpError` instead of silently running unwarmed --
       off by default so a scripted/no-model client (which has nothing to
       warm up) never has to think about it, and so ad-hoc REPL use isn't
       penalized.
    2. **Every returned action is schema-validated** via
       `validate_env_action` before it reaches the caller.
    """

    #: Class- or instance-level default; override per instance via the
    #: `control_contract=` constructor kwarg (e.g. to test against a
    #: non-default contract).
    control_contract: ControlContract = FROZEN_CONTROL_CONTRACT

    def __init__(self, *, strict_warmup: bool = False,
                 control_contract: "ControlContract | None" = None):
        self._warmed_up = False
        self._strict_warmup = strict_warmup
        if control_contract is not None:
            self.control_contract = control_contract

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    def warmup(self, obs: Mapping, prompt: str = "warmup") -> None:
        """Run one throwaway inference to pay for first-call cost (JAX jit
        for served models; a no-op for scripted ones) before any timed
        rollout. Required before `__call__` when `strict_warmup=True`;
        recommended unconditionally regardless (see class docstring)."""
        self._warmup_impl(obs, prompt)
        self._warmed_up = True

    @abc.abstractmethod
    def _warmup_impl(self, obs: Mapping, prompt: str) -> None:
        """Subclass hook: do the actual warmup inference (or nothing)."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear any per-episode state (action-chunk cache, step counters,
        sinusoid phase, ...). Called once per episode, before the first
        `__call__`."""

    @abc.abstractmethod
    def _act(self, obs: Mapping, prompt: str) -> "np.ndarray | Sequence[float]":
        """Subclass hook: return the raw env-facing action for one control
        tick (validated by the public `__call__` wrapper)."""

    def __call__(self, obs: Mapping, prompt: str) -> np.ndarray:
        if self._strict_warmup and not self._warmed_up:
            raise PolicyNotWarmedUpError(
                f"{type(self).__name__}.__call__ invoked before warmup(); "
                "call warmup(obs, prompt) once after connecting/loading and "
                "before any timed rollout (docs/ROBOT.md JAX-jit gotcha)")
        return validate_env_action(self._act(obs, prompt), self.control_contract)


# ------------------------------------------------- deterministic replay ---

def run_trace(
    client: PolicyClient,
    observations: Sequence[Mapping],
    prompt: str,
    *,
    warmup_obs: "Mapping | None" = None,
) -> list[np.ndarray]:
    """Deterministic observation -> action trace replay (Task 08 step 5).

    Feeds `observations` through `client` in order -- `client.reset()`
    first, then an optional `warmup()` if `warmup_obs` is given and the
    client isn't already warmed up -- and returns the list of validated
    env-facing actions.

    Determinism is a property of the underlying client, not of this
    function: `ScriptedPolicyClient` is deterministic by construction, and
    `Pi05PolicyClient` is deterministic when wired to a fixed-response
    `connector` (see tests/test_control_contract.py and
    robo/policy/clients/pi05_client.py), which is exactly what makes this
    usable for golden-trace tests without a live openpi server.

    This is the intended reuse point for other run-time consumers (e.g. the
    paired rollout runner, robo/eval/paired_runner.py) that want a policy's
    action trace against a fixed observation sequence without re-deriving
    the per-client call protocol themselves.
    """
    if warmup_obs is not None and not client.warmed_up:
        client.warmup(warmup_obs, prompt)
    client.reset()
    return [client(obs, prompt) for obs in observations]


__all__ = [
    "ACTION_DIM",
    "JOINT_DIM",
    "FROZEN_FIELDS_PATH",
    "ControlContract",
    "FROZEN_CONTROL_CONTRACT",
    "load_frozen_control_contract",
    "ControlContractError",
    "ControlContractMismatchError",
    "PolicyNotWarmedUpError",
    "EnvActionShapeError",
    "validate_against_frozen",
    "assert_matches_frozen",
    "validate_env_action",
    "PolicyClient",
    "run_trace",
]
