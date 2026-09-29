"""robo.policy.clients.pi05_client: PolicyClient adapter around the openpi
websocket server, porting robo/eval/pi05_eval.py::ServerPolicy onto the
shared robo.policy.control_contract.PolicyClient interface.

**This module does not import robo/eval/pi05_eval.py** (owned by a parallel
task; it may still be edited concurrently). The wire contract below is
ported from reading that file plus run/pi05_serve.sh and docs/ROBOT.md as
references, not reinvented:

  - request dict: `{"observation/exterior_image_1_left": uint8 HWC resized
    to image_resize_hw via openpi_client.image_tools.resize_with_pad,
    "observation/wrist_image_left": same, "observation/joint_position":
    float (7,), "observation/gripper_position": float (1,), "prompt": str}`
  - response dict: `{"actions": float array, shape (>=chunk_size, >=8)}` --
    a chunk of **absolute** joint-position targets (7) + gripper [0, 1].
    The model predicts joint *deltas* internally; openpi's server-side
    `AbsoluteActions` transform (docs/ROBOT.md) converts to this absolute
    chunk before it ever reaches the client, so nothing downstream of this
    file (including robo/envs/pi05_env.py) ever sees a delta.
  - the client executes the full `chunk_size` (15) steps of one chunk at
    15 Hz, then requeries -- "RoboLab default" per docs/ROBOT.md.
  - first inference triggers a JAX jit (minutes); the websocket's 20s
    keepalive kills the connection meanwhile, so both `ServerPolicy` and
    this port reconnect-and-retry around it (retry cap 5, per docs/ROBOT.md:
    "do not lower it").

Both real checkpoints in the registry (`pi05_droid_jointpos` and
`droid_pi05_jointpos_with_web_and_sim`) share this exact wire contract --
run/pi05_serve.sh only changes `--policy.config`/`--policy.dir` server-side
to switch between them.

Test/replay support: the real `openpi_client.websocket_client_policy.
WebsocketClientPolicy` is only importable/reachable with a live server on
the network. For deterministic trace replays (no server, no network),
pass `connector=some_callable` at
construction; `some_callable()` must return an object exposing
`.infer(request_dict) -> {"actions": ndarray}`, exactly the
`WebsocketClientPolicy.infer` surface. This is the one seam this module
adds beyond a literal port, and it is what makes `robo.policy.
control_contract.run_trace` usable against this client without a server.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from robo.policy.control_contract import ControlContract, PolicyClient
from robo.policy.runtime_identity import (
    PolicyRuntimeIdentityError,
    identity_from_server_metadata,
)

# () -> object with .infer(request: dict) -> {"actions": ndarray}, matching
# openpi_client.websocket_client_policy.WebsocketClientPolicy's surface.
Connector = Callable[[], object]


def _resize_with_pad(image, height: int, width: int):
    # Lazy import: openpi_client is only needed on the real serving path,
    # not for tests that inject a `connector` and never touch real images
    # through the network client -- but many tests DO exercise this exact
    # resize step against synthetic images, since it's part of the real
    # contract being ported, not an implementation detail to mock away.
    from openpi_client import image_tools
    return image_tools.resize_with_pad(image, height, width)


class Pi05PolicyClient(PolicyClient):
    """Adapter for the pi05_droid_jointpos-family openpi websocket server.
    Shared by both real checkpoints in the registry
    (`pi05_droid_jointpos`, `droid_pi05_jointpos_with_web_and_sim`) --
    see module docstring."""

    def __init__(self, host: str = "localhost", port: int = 8000,
                 open_loop_horizon: int = 15,
                 connector: "Connector | None" = None,
                 retries: int = 5, retry_sleep_s: float = 10.0,
                 strict_warmup: bool = False,
                 control_contract: "ControlContract | None" = None,
                 expected_server_identity: "Mapping | None" = None,
                 expected_policy_identity: "Mapping | None" = None):
        super().__init__(strict_warmup=strict_warmup, control_contract=control_contract)
        self._host, self._port = host, port
        self.horizon = open_loop_horizon
        self._connector = connector
        self._retries = retries
        self._retry_sleep_s = retry_sleep_s
        self._client = None
        self._chunk, self._i = None, 0
        self._expected_server_identity = (
            copy.deepcopy(dict(expected_server_identity))
            if expected_server_identity is not None else None)
        self._expected_policy_identity = (
            copy.deepcopy(dict(expected_policy_identity))
            if expected_policy_identity is not None else None)
        self._verified_server_identity = None
        self._sampling = (copy.deepcopy(self._expected_server_identity["policy"].get("sampling"))
            if self._expected_server_identity is not None else None)
        if self._sampling is not None:
            from robo.policy.sampling_contract import validate_contract
            validate_contract(self._sampling, self._expected_server_identity['policy']['training_config'])
            if (isinstance(open_loop_horizon, bool) or not isinstance(open_loop_horizon, int)
                    or open_loop_horizon != self._sampling['shape'][0]):
                raise ValueError('deterministic sampling requires the frozen 15-step chunk horizon')
        self._sampling_seed = None
        self._sampling_chunk = 0
        self._sampling_receipts = []

    @classmethod
    def from_entry(cls, entry, *, host: str = "localhost", port: int = 8000,
                   connector: "Connector | None" = None,
                   expected_server_identity: "Mapping | None" = None,
                   **kwargs) -> "Pi05PolicyClient":
        """Build from a `robo.policy.registry.PolicyEntry`, honoring its
        `chunk_size` (open_loop_horizon) and image resize contract via
        `entry`'s already-validated control contract fields."""
        if not entry.checkpoint_path or not entry.checkpoint_hash:
            raise PolicyRuntimeIdentityError(
                f"policy {entry.id!r} lacks a bound checkpoint path/hash")
        if not entry.training_config:
            raise PolicyRuntimeIdentityError(
                f"policy {entry.id!r} lacks a bound OpenPI training config")
        expected_policy_identity = {
            "id": entry.id,
            "checkpoint_path": str(Path(
                entry.checkpoint_path).expanduser().resolve(strict=False)),
            "checkpoint_fingerprint": entry.checkpoint_hash,
            "training_config": entry.training_config,
        }
        return cls(
            host=host, port=port, open_loop_horizon=entry.chunk_size,
            connector=connector,
            expected_server_identity=expected_server_identity,
            expected_policy_identity=expected_policy_identity,
            **kwargs)

    # --------------------------------------------------------- connection --

    def _connect(self):
        if self._connector is not None:
            client = self._connector()
        else:
            from openpi_client import websocket_client_policy
            client = websocket_client_policy.WebsocketClientPolicy(
                host=self._host, port=self._port)
        # A registry-created real client always has expected_policy_identity.
        # It must never send an observation to an unidentifiable or mismatched
        # server.  Directly-constructed clients retain their existing mock/test
        # seam unless an expected identity was explicitly supplied.
        if (self._expected_policy_identity is not None
                or self._expected_server_identity is not None):
            getter = getattr(client, "get_server_metadata", None)
            if not callable(getter):
                raise PolicyRuntimeIdentityError(
                    "OpenPI client does not expose websocket server metadata")
            self._verified_server_identity = identity_from_server_metadata(
                getter(), expected=self._expected_server_identity,
                expected_policy=self._expected_policy_identity)
        return client

    def verify_server_identity(self) -> dict:
        """Connect and authenticate server identity without running inference."""
        if self._client is None:
            self._client = self._connect()
        if self._verified_server_identity is None:
            raise PolicyRuntimeIdentityError(
                "policy client has no verified server runtime identity")
        return copy.deepcopy(self._verified_server_identity)

    @property
    def verified_server_identity(self) -> "dict | None":
        return (copy.deepcopy(self._verified_server_identity)
                if self._verified_server_identity is not None else None)

    def _infer_retry(self, req: dict) -> dict:
        # Ported from robo/eval/pi05_eval.py::ServerPolicy._infer_retry
        # (same reasoning as RoboLab's client-side _infer_with_retry): first
        # inference triggers a JAX jit (minutes); the websockets 20s
        # keepalive kills the connection meanwhile -> reconnect and resend.
        if self._client is None:
            self._client = self._connect()
        last_exc = None
        for k in range(self._retries):
            try:
                return self._client.infer(req)
            except Exception as e:  # noqa: BLE001 -- retried below, re-raised at the end
                last_exc = e
                if k == self._retries - 1:
                    raise
                print(f"[pi05_client] infer failed ({type(e).__name__}); "
                      f"reconnecting ({k + 1}/{self._retries})", flush=True)
                time.sleep(self._retry_sleep_s)
                try:
                    self._client = self._connect()
                except PolicyRuntimeIdentityError:
                    # A replacement server with different code/checkpoint is
                    # not a transient network failure and must never be hidden.
                    raise
                except Exception:
                    pass
        raise last_exc  # pragma: no cover -- loop always returns or raises above

    def _request(self, obs: Mapping, prompt: str) -> dict:
        h, w = self.control_contract.image_resize_hw
        return {
            "observation/exterior_image_1_left": _resize_with_pad(
                obs["observation/exterior_image_1_left"], h, w),
            "observation/wrist_image_left": _resize_with_pad(
                obs["observation/wrist_image_left"], h, w),
            "observation/joint_position": obs["observation/joint_position"],
            "observation/gripper_position": obs["observation/gripper_position"],
            "prompt": prompt,
        }

    # -------------------------------------------------- PolicyClient API --

    def _sampled_infer(self, obs: Mapping, prompt: str, *, domain: str) -> dict:
        from robo.policy.sampling_contract import REQUEST_KEY, RECEIPT_KEY, request, noise_and_receipt
        req = self._request(obs, prompt)
        expected = None
        if self._sampling is not None:
            if domain == 'episode' and self._sampling_seed is None:
                raise PolicyRuntimeIdentityError('paired policy requires begin_episode(seed) before actions')
            envelope = request(self._sampling,
                seed=0 if domain == 'warmup' else self._sampling_seed,
                chunk=0 if domain == 'warmup' else self._sampling_chunk, domain=domain)
            req[REQUEST_KEY] = envelope
            _, expected = noise_and_receipt(envelope, self._sampling)
        result = self._infer_retry(req)
        if expected is not None:
            if result.get(RECEIPT_KEY) != expected:
                raise PolicyRuntimeIdentityError('server sampling receipt differs or is missing')
            if domain == 'episode':
                self._sampling_receipts.append(copy.deepcopy(expected))
                self._sampling_chunk += 1
        return result

    def _warmup_impl(self, obs: Mapping, prompt: str) -> None:
        self._sampled_infer(obs, prompt, domain='warmup')

    def begin_episode(self, seed: int) -> None:
        from robo.policy.sampling_contract import request
        self.reset()
        if self._sampling is not None:
            request(self._sampling, seed=seed, chunk=0, domain='episode')
            self._sampling_seed = seed

    @property
    def sampling_receipts(self) -> list:
        return copy.deepcopy(self._sampling_receipts)


    def reset(self) -> None:
        self._chunk, self._i = None, 0
        self._sampling_seed = None
        self._sampling_chunk = 0
        self._sampling_receipts = []

    def _act(self, obs: Mapping, prompt: str) -> np.ndarray:
        if self._chunk is None or self._i >= min(self.horizon, len(self._chunk)):
            res = self._sampled_infer(obs, prompt, domain="episode")
            chunk = np.asarray(res["actions"])
            action_dim = self.control_contract.action_dim
            if chunk.ndim != 2 or chunk.shape[1] < action_dim:
                raise ValueError(
                    f"unexpected action chunk shape {chunk.shape} (expected "
                    f"(>=1, >={action_dim}))")
            if self._sampling is not None and chunk.shape[0] != self._sampling["shape"][0]:
                raise ValueError("sampled action chunk must contain exactly 15 rows")
            self._chunk = chunk
            self._i = 0
        a = self._chunk[self._i, :self.control_contract.action_dim]
        self._i += 1
        return a


__all__ = ["Pi05PolicyClient", "Connector"]
