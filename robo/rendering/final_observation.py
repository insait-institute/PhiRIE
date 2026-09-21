"""Transactional same-state observation bridge for the final native experiment.

The existing simulator owns dynamics and the existing harness owns rollouts.
Call this bridge when assembling policy images. Rendering and enhancement are
injected from the admitted GS/Harmonizer backends, never synthetic fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    episode: str
    tick: int
    state_sha256: str


@dataclass
class Packet:
    snapshot: Snapshot
    rgb: np.ndarray
    visible_robot_core: np.ndarray
    buffers: dict


class StateBoundObserver:
    """All cameras commit together; repeats do not advance temporal history.

    render(camera, snapshot) -> Packet
    enhance(camera, raw, previous_emitted_or_None, buffers) -> uint8 RGB
    constrain(camera, raw, enhanced, previous_emitted_or_None, buffers) -> RGB

    Callback buffers come from the reconstructed scene. The capture adapter must
    hash all policy-relevant state in Snapshot, including camera/object transforms.
    This bridge checks synchronization, not the correctness of those transforms.
    """
    MODES = {"gaussian", "official_h", "robot_restore_h", "state_h"}

    def __init__(self, cameras: list[str], snapshot: Callable[[], Snapshot],
                 render: Callable[[str, Snapshot], Packet], *, mode: str,
                 enhance: Callable | None = None, constrain: Callable | None = None):
        if not cameras or len(set(cameras)) != len(cameras) or mode not in self.MODES:
            raise ValueError("declare unique policy cameras and a supported mode")
        if mode != "gaussian" and enhance is None:
            raise ValueError("real enhancer is required; no implicit identity fallback")
        if mode == "state_h" and constrain is None:
            raise ValueError("state-conditioned arm needs the actual correction function")
        self.cameras = tuple(cameras)
        self.snapshot, self.render = snapshot, render
        self.mode, self.enhance, self.constrain = mode, enhance, constrain
        self.episode = None
        self.previous: dict[str, np.ndarray] = {}
        self.cached_snapshot = None
        self.cache: dict[str, np.ndarray] = {}
        self.events: list[dict] = []

    @staticmethod
    def _rgb(array: np.ndarray, shape: tuple | None = None) -> np.ndarray:
        a = np.asarray(array)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
            raise ValueError("backend must explicitly convert to HxWx3 uint8 RGB")
        if shape is not None and a.shape != shape:
            raise ValueError("enhancement changed the frozen canvas")
        return a.copy()

    def observe(self) -> dict[str, np.ndarray]:
        before = self.snapshot()
        if not before.episode or before.tick < 0 or not before.state_sha256:
            raise ValueError("invalid current simulator snapshot")
        if before.episode != self.episode:
            self.episode = before.episode
            self.previous, self.cache, self.cached_snapshot = {}, {}, None
        if before == self.cached_snapshot:
            return {k: v.copy() for k, v in self.cache.items()}
        if self.cached_snapshot is not None and before.tick <= self.cached_snapshot.tick:
            raise ValueError("state changed at the same/backward tick; use a new episode ID for reset")
        started = perf_counter()
        candidate = {}
        try:
            for camera in self.cameras:
                packet = self.render(camera, before)
                if packet.snapshot != before:
                    raise ValueError("renderer returned stale or cross-camera state")
                raw = self._rgb(packet.rgb)
                previous = self.previous.get(camera)
                previous = None if previous is None else previous.copy()
                if self.mode == "gaussian":
                    image = raw.copy()
                else:
                    image = self._rgb(self.enhance(camera, raw.copy(), previous, packet.buffers), raw.shape)
                    if self.mode == "state_h":
                        image = self._rgb(self.constrain(camera, raw.copy(), image, previous, packet.buffers), raw.shape)
                    if self.mode in ("robot_restore_h", "state_h"):
                        mask = np.asarray(packet.visible_robot_core)
                        if mask.dtype != bool or mask.shape != raw.shape[:2]:
                            raise ValueError("visible robot CORE must be a boolean same-canvas mask")
                        image[mask] = raw[mask]
                candidate[camera] = image
            if self.snapshot() != before:
                raise ValueError("physics advanced while synchronous observation was being produced")
        except Exception as exc:
            self.events.append({"episode": before.episode, "tick": before.tick,
                                "status": "OBSERVATION_FAILED", "error_type": type(exc).__name__,
                                "wall_s": perf_counter()-started, "fallback": False})
            # No partial camera/history commit and no substitution of raw images.
            raise
        self.previous = {k: v.copy() for k, v in candidate.items()}
        self.cache = {k: v.copy() for k, v in candidate.items()}
        self.cached_snapshot = before
        self.events.append({"episode": before.episode, "tick": before.tick,
                            "state_sha256": before.state_sha256, "cameras": list(self.cameras),
                            "status": "CURRENT_STATE", "wall_s": perf_counter()-started,
                            "max_state_lag_ticks": 0, "fallback": False})
        return candidate

    def summary(self) -> dict:
        valid = [e for e in self.events if e["status"] == "CURRENT_STATE"]
        times = [e["wall_s"] for e in valid]
        return {"mode": self.mode, "successful_observation_requests": len(valid),
                "failed_observation_requests": len(self.events)-len(valid),
                "camera_frames": sum(len(e["cameras"]) for e in valid),
                "p95_request_s": float(np.percentile(times, 95)) if times else None,
                "silent_fallback_frames": 0,
                "limitations": "synchronization and declared core exactness; not geometric correctness"}


class PolicyObservationAdapter:
    """Replace only policy images at RoboCasaAdapter.get_policy_observation().

    The admitted backend provides the exact camera-to-observation-key mapping
    and uses the SAME native image preprocessing. All dynamics, actions, state,
    success and stage reporting are delegated unchanged to the native adapter.
    """
    def __init__(self, native_adapter, observer: StateBoundObserver,
                 camera_keys: Mapping[str, str], preprocess: Callable[[str, np.ndarray], np.ndarray]):
        if set(camera_keys) != set(observer.cameras) or len(set(camera_keys.values())) != len(camera_keys):
            raise ValueError("one explicit native observation key per policy camera")
        self._native_adapter, self._observer = native_adapter, observer
        self._camera_keys, self._preprocess = dict(camera_keys), preprocess

    def __getattr__(self, name):
        return getattr(self._native_adapter, name)

    def get_policy_observation(self):
        original = self._native_adapter.get_policy_observation()
        if not isinstance(original, dict):
            raise ValueError("native observation must be a dictionary")
        result = dict(original)
        emitted = self._observer.observe()
        for camera, key in self._camera_keys.items():
            if key not in original:
                raise ValueError("camera key does not exist in admitted native observations")
            image = np.asarray(self._preprocess(camera, emitted[camera]))
            native_image = np.asarray(original[key])
            if image.shape != native_image.shape or image.dtype != native_image.dtype:
                raise ValueError("GS preprocessing differs from the native observation contract")
            if not np.isfinite(image).all():
                raise ValueError("nonfinite policy image")
            result[key] = image.copy()
        return result
