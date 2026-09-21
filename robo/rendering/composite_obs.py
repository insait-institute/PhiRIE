"""Stable adapter exposing the repository's existing photoreal CompositeObs.

The original observer lives in ``robo.eval.pi05_eval`` and predates the paper
harness. This module gives the harness one public factory while tolerating the
legacy constructor's historical argument names. It does not implement a second
renderer.
"""
from __future__ import annotations

import inspect
from pathlib import Path


class CompositeAdapterError(RuntimeError):
    pass


def _candidate_class():
    from robo.eval import pi05_eval
    for name in ("CompositeObs", "CompositeObservation", "PhotorealObs"):
        value = getattr(pi05_eval, name, None)
        if value is not None:
            return value
    # The maintained renderer lives in ``robo.rendering.pi05_render``.  Older
    # revisions re-exported it from ``pi05_eval``; current ones instantiate it
    # locally inside the legacy CLI instead.  Import the maintained class
    # directly so the paper harness does not depend on that historical alias.
    from robo.rendering.pi05_render import CompositeObs

    return CompositeObs


def _keyword_pool(*, env, suite, factory_dir, render_wh, options):
    factory_dir = Path(factory_dir)
    width, height = tuple(render_wh)
    return {
        "env": env, "environment": env, "sim_env": env,
        "suite": suite, "task_suite": suite, "scene_cfg": suite,
        "factory_dir": factory_dir, "out_dir": factory_dir,
        "output_dir": factory_dir, "scene_dir": factory_dir,
        "scene_root": factory_dir, "root": factory_dir,
        "render_wh": (width, height), "resolution": (width, height),
        "width": width, "height": height, "W": width, "H": height,
        "options": options,
    }


def _construct(cls, pool):
    signature = inspect.signature(cls)
    kwargs = {name: pool[name] for name in signature.parameters if name in pool}
    missing = [name for name, parameter in signature.parameters.items()
               if name not in kwargs
               and parameter.default is inspect.Parameter.empty
               and parameter.kind in {inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY}]
    if not missing:
        return cls(**kwargs)

    # Historical positional layouts used by the project. Each attempt still
    # reuses the original class; it never changes rendering semantics.
    env, suite, factory_dir = pool["env"], pool["suite"], pool["factory_dir"]
    candidates = [
        (env, factory_dir, suite), (env, suite, factory_dir),
        (env, factory_dir), (env, suite), (factory_dir, env),
        (factory_dir,), (env,),
    ]
    errors = []
    for args in candidates:
        try:
            return cls(*args)
        except TypeError as exc:
            errors.append(str(exc))
    raise CompositeAdapterError(
        f"could not instantiate {cls} with required parameters {missing}; "
        + " | ".join(errors[-3:]))


def build_observer(*, env, suite, factory_dir=None, out_dir=None,
                   render_wh=(640, 360), options=None, **_):
    """Construct the existing CompositeObs through one harness-facing API."""
    directory = factory_dir if factory_dir is not None else out_dir
    if directory is None:
        raise CompositeAdapterError("factory_dir/out_dir is required")
    options = dict(options or {})
    cls = _candidate_class()
    return _construct(cls, _keyword_pool(
        env=env, suite=suite, factory_dir=directory,
        render_wh=render_wh, options=options))
