"""Official RoboCasa OpenPI observation contract with paired chunk RNG.

No simulator state or object oracle enters the policy. The server owns the
unchanged official checkpoint, normalization transforms, and action sampler.
"""
from __future__ import annotations

from collections import deque
import hashlib
from typing import Any

import numpy as np

POLICY_CONFIG = "pi05_pretrain_human300"
UPSTREAM_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
CHECKPOINT_REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
CAMERA_KEYS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}


def chunk_seed(episode_seed: int, chunk_index: int, *, instance_id=None, reset_id=None) -> int:
    """Stable seed; callers bind episode seed to the paired task/reset identity."""
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
        raise ValueError("episode seed must be a nonnegative integer")
    if isinstance(chunk_index,bool) or not isinstance(chunk_index,int) or chunk_index<0:
        raise ValueError('chunk index must be nonnegative')
    if instance_id is None and reset_id is None:
        payload = f"simany-sr0-policy-v1/{episode_seed}/{chunk_index}".encode()
    else:
        import json
        if not isinstance(instance_id,str) or not instance_id or not isinstance(reset_id,str) or not reset_id:
            raise ValueError('v2 RNG requires canonical instance and reset IDs')
        payload=json.dumps(['simany-native-policy-v2',instance_id,reset_id,episode_seed,chunk_index],
                           separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def pack_observation(observation: dict[str, Any]) -> dict[str, Any]:
    from openpi_client import image_tools

    result = {}
    for destination, source in CAMERA_KEYS.items():
        image = np.asarray(observation[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{source} must be HWC uint8 RGB")
        result[destination] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(image), 224, 224)
        )
    parts = [np.asarray(observation[key]) for key in STATE_KEYS]
    if [part.shape for part in parts] != [(3,), (4,), (3,), (4,), (2,)]:
        raise ValueError("RoboCasa PandaOmron proprioception must have dimensions 3,4,3,4,2")
    state = np.concatenate(parts, axis=0)
    if not np.isfinite(state).all():
        raise ValueError("nonfinite native proprioception")
    prompt = observation["annotation.human.task_description"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("native task description must be a nonempty string")
    result.update({"observation/state": state, "prompt": prompt})
    return result


class NativePolicy:
    is_visual_policy = True

    def __init__(self, host: str, port: int, replan_steps: int = 5, *, client=None):
        if replan_steps != 5:
            raise ValueError("this native policy contract freezes official replan_steps=5")
        if client is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy(host=host, port=port)
        self._client = client
        self._metadata = dict(client.get_server_metadata())
        if self._metadata.get("policy_config") != POLICY_CONFIG:
            raise ValueError("server is not the declared RoboCasa checkpoint/config")
        if self._metadata.get("rng_protocol") != "simany-sr0-policy-v1/per-chunk-jax-key":
            raise ValueError("server lacks deterministic per-chunk RNG binding")
        if self._metadata.get("openpi_commit") != UPSTREAM_COMMIT:
            raise ValueError("server OpenPI source pin differs")
        if self._metadata.get("checkpoint_revision") != CHECKPOINT_REVISION:
            raise ValueError("server checkpoint revision differs")
        if not self._metadata.get("checkpoint_receipt_sha256"):
            raise ValueError("server has no checkpoint content receipt")
        self._replan_steps = replan_steps
        self._queue = deque()
        self._seed = None
        self._chunk_index = 0
        self.last_response_metadata = None
        self._stream_identity = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._metadata, "replan_steps": self._replan_steps,
                "policy_state_keys": list(STATE_KEYS), "policy_camera_keys": dict(CAMERA_KEYS)}

    def bind_stream(self, *, canonical_instance_id, reset_id, policy_rng_seed):
        """Bind identity before the canonical harness calls reset(seed)."""
        chunk_seed(policy_rng_seed,0,instance_id=canonical_instance_id,reset_id=reset_id)
        self._stream_identity={'canonical_instance_id':canonical_instance_id,'reset_id':reset_id,
                               'policy_rng_seed':policy_rng_seed}

    @property
    def stream_identity(self):
        return None if self._stream_identity is None else dict(self._stream_identity)

    def state_dict(self):
        return {'schema_version':1,'seed':self._seed,'chunk_index':self._chunk_index,
                'stream_identity':self.stream_identity,'pending_actions':[a.tolist() for a in self._queue],
                'checkpoint_receipt_sha256':self._metadata['checkpoint_receipt_sha256']}

    def load_state_dict(self, state):
        """Restore an exact client queue/chunk boundary after reconnect."""
        if state.get('schema_version')!=1 or state.get('checkpoint_receipt_sha256')!=self._metadata['checkpoint_receipt_sha256']:
            raise ValueError('resumed client checkpoint/state differs')
        identity=state.get('stream_identity')
        if identity is not None:
            self.bind_stream(**identity)
        else:self._stream_identity=None
        self.reset(state['seed'])
        chunk_seed(self._seed,state['chunk_index'])
        pending=np.asarray(state['pending_actions'],dtype=float)
        if pending.size==0:pending=pending.reshape(0,12)
        if pending.ndim!=2 or pending.shape[1]!=12 or len(pending)>5 or not np.isfinite(pending).all():
            raise ValueError('invalid resumed action queue')
        self._chunk_index=state['chunk_index'];self._queue=deque(a.copy() for a in pending)

    def reset(self, seed: int) -> None:
        if self._stream_identity is not None:
            seed=self._stream_identity['policy_rng_seed']
        chunk_seed(seed, 0)  # validates seed without model inference
        self._seed = seed
        self._chunk_index = 0
        self._queue.clear()
        self.last_response_metadata = None

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        if self._seed is None:
            raise RuntimeError("reset(seed) is required before inference")
        if not self._queue:
            element = pack_observation(observation)
            binding=self._stream_identity
            seed = chunk_seed(self._seed, self._chunk_index,
                instance_id=None if binding is None else binding['canonical_instance_id'],
                reset_id=None if binding is None else binding['reset_id'])
            element["_simany_rng_seed"] = seed
            response = self._client.infer(element)
            if response.get("simany_rng_seed") != seed:
                raise ValueError("server RNG receipt does not match paired chunk")
            actions = np.asarray(response["actions"])
            if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) < self._replan_steps:
                raise ValueError("official RoboCasa action chunk must be N x 12, N >= 5")
            if not np.isfinite(actions).all():
                raise ValueError("nonfinite policy action")
            self._queue.extend(np.array(a, copy=True) for a in actions[:self._replan_steps])
            self.last_response_metadata = {key: value for key, value in response.items() if key != "actions"}
            self._chunk_index += 1
        return self._queue.popleft()

    def close(self) -> None:
        connection = getattr(self._client, "_ws", None)
        if connection is not None:
            connection.close()
