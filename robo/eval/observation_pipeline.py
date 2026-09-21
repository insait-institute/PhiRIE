"""Observation interventions for the paired harness.

Raster observations require no adapter. Composite observations reuse the
repository observer when available, or a dotted factory declared in YAML.
"""
from __future__ import annotations

import importlib
import copy
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from robo.rendering.harmonizer_client import EnhancerError, EnhancerProtocol
from robo.rendering.robot_restore import restore_robot_core

DEFAULT_IMAGE_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)


class ObservationError(RuntimeError):
    pass


@dataclass
class ObservationResult:
    observation: dict[str, Any]
    latency_ms: float
    per_camera: dict[str, dict] = field(default_factory=dict)


class ObservationSource:
    def reset(self) -> None:
        return None

    def get_obs(self) -> dict[str, Any]:
        raise NotImplementedError


class RasterSource(ObservationSource):
    def __init__(self, env):
        self.env = env

    def get_obs(self) -> dict[str, Any]:
        return self.env.get_obs()


class ObjectSource(ObservationSource):
    def __init__(self, obj):
        self.obj = obj

    def reset(self) -> None:
        reset = getattr(self.obj, "reset", None)
        if callable(reset):
            reset()

    def get_obs(self) -> dict[str, Any]:
        for name in ("get_obs", "observation", "render_obs"):
            method = getattr(self.obj, name, None)
            if callable(method):
                value = method()
                if isinstance(value, dict):
                    return value
        if callable(self.obj):
            value = self.obj()
            if isinstance(value, dict):
                return value
        raise ObservationError(
            f"composite observer {type(self.obj).__name__} exposes no dict-valued get_obs")


def _import_symbol(path: str):
    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        module_name, separator, symbol_name = path.rpartition(".")
    if not module_name or not symbol_name:
        raise ValueError(f"expected dotted or module:symbol path, got {path!r}")
    return getattr(importlib.import_module(module_name), symbol_name)


def _call_with_supported_kwargs(factory: Callable, kwargs: dict[str, Any]):
    signature = inspect.signature(factory)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return factory(**kwargs)
    accepted = {name: value for name, value in kwargs.items()
                if name in signature.parameters}
    required = [name for name, parameter in signature.parameters.items()
                if parameter.default is parameter.empty
                and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD,
                                       parameter.KEYWORD_ONLY}
                and name not in accepted]
    if required:
        raise TypeError(f"observer factory {factory} requires unsupported args {required}")
    return factory(**accepted)


def build_source(env, *, suite: dict, factory_dir, options: dict,
                 observation_variant: str) -> ObservationSource:
    if observation_variant == "raster":
        return RasterSource(env)
    kwargs = {
        "env": env, "suite": suite, "factory_dir": factory_dir,
        "out_dir": factory_dir, "options": options,
        "render_wh": tuple(options.get("render_wh", (640, 360))),
    }
    candidates = []
    if options.get("source_factory"):
        candidates.append(options["source_factory"])
    candidates.extend([
        "robo.eval.pi05_eval:build_composite_observer",
        "robo.eval.pi05_eval:CompositeObs",
        "robo.rendering.composite_obs:CompositeObs",
        "robo.rendering.composite:CompositeObs",
    ])
    errors = []
    for candidate in candidates:
        try:
            factory = _import_symbol(candidate)
            return ObjectSource(_call_with_supported_kwargs(factory, kwargs))
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise ObservationError(
        "no usable composite observation source; set options.source_factory. "
        + " | ".join(errors))


def _mask_candidates(image_key: str) -> tuple[str, ...]:
    return (image_key + "_robot_mask", image_key.replace("image", "robot_mask"),
            image_key.replace("_left", "_robot_mask"))


def get_robot_mask(env, obs: dict, image_key: str) -> np.ndarray:
    for key in _mask_candidates(image_key):
        if key in obs:
            return np.asarray(obs[key])
    camera_name = image_key.split("/")[-1].replace("_image", "")
    for method_name in ("render_robot_mask", "get_robot_mask"):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                return np.asarray(method(camera_name))
            except TypeError:
                return np.asarray(method(image_key))
    raise ObservationError(
        f"Option C requires an exact robot mask for {image_key}; none was exposed")


class ObservationPipeline:
    def __init__(self, source: ObservationSource, *, variant: str, env,
                 enhancer: EnhancerProtocol | None = None,
                 image_keys=DEFAULT_IMAGE_KEYS, restoration_options=None,
                 color_match_calibrations=None):
        self.source = source
        self.variant = variant
        self.env = env
        self.enhancer = enhancer
        self.image_keys = tuple(image_keys)
        self.restoration_options = dict(restoration_options or {})
        self.color_match_calibrations = copy.deepcopy(dict(color_match_calibrations or {}))
        if variant == "color_match" and set(self.color_match_calibrations) != set(self.image_keys):
            raise ObservationError("color_match requires one frozen calibration for each camera")
        self.episode_id: str | None = None
        self.frame_index = 0

    def reset(self, episode_id: str) -> None:
        self.source.reset()
        self.episode_id = episode_id
        self.frame_index = 0
        if self.variant == "harmonizer_c":
            if self.enhancer is None:
                raise ObservationError("harmonizer_c requires an enhancer client")
            for image_key in self.image_keys:
                self.enhancer.reset_stream(self._stream_id(image_key))

    def _stream_id(self, image_key: str) -> str:
        if self.episode_id is None:
            raise ObservationError("observation pipeline was not reset for this episode")
        return f"{self.episode_id}/{image_key}"

    def get_obs(self) -> ObservationResult:
        start = time.perf_counter()
        obs = dict(self.source.get_obs())
        per_camera: dict[str, dict] = {}
        if self.variant == "color_match":
            from robo.rendering.color_match import apply_background_affine, ALGORITHM
            for image_key in self.image_keys:
                if image_key not in obs:
                    raise ObservationError(f"missing color-match image {image_key}")
                camera_start = time.perf_counter()
                calibration = self.color_match_calibrations[image_key]
                obs[image_key] = apply_background_affine(obs[image_key], calibration, camera_id=image_key)
                per_camera[image_key] = {
                    "backend": "color_match", "algorithm": ALGORITHM,
                    "calibration_sha256": calibration["calibration_sha256"],
                    "transform_latency_ms": (time.perf_counter() - camera_start) * 1000.0}
            self.frame_index += 1
            return ObservationResult(obs, (time.perf_counter() - start) * 1000.0, per_camera)
        if self.variant != "harmonizer_c":
            return ObservationResult(obs, (time.perf_counter() - start) * 1000.0)
        assert self.enhancer is not None
        for image_key in self.image_keys:
            if image_key not in obs:
                raise ObservationError(f"missing policy image {image_key}")
            raw = np.asarray(obs[image_key])
            try:
                enhanced, metadata = self.enhancer.enhance(
                    raw, stream_id=self._stream_id(image_key),
                    frame_index=self.frame_index)
            except EnhancerError:
                raise
            mask = get_robot_mask(self.env, obs, image_key)
            final, stats = restore_robot_core(raw, enhanced, mask,
                                              **self.restoration_options)
            obs[image_key] = final
            per_camera[image_key] = {
                **metadata, "robot_core_equal": stats.core_equal,
                "max_robot_core_error": stats.max_core_error,
                "robot_core_pixels": stats.core_pixels}
        self.frame_index += 1
        return ObservationResult(
            obs, (time.perf_counter() - start) * 1000.0, per_camera)
