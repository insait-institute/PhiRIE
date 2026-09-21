"""Hierarchical bootstrap for the oracle_causal protocol (Task 10,
plan/10_PREDICTIVE_METRICS_AND_POWER.md step 2:
"Use hierarchical bootstrap over scenes/tasks, then paired seeds; publish
intervals and samples.").

The resampling unit hierarchy is scene -> seed. A "seed" bucket for a given
scene holds every record sharing that (scene, seed) pair -- typically one
record per policy being compared, so the pairing across policies for a fixed
episode seed is preserved automatically: resampling never splits a seed
bucket apart, it only resamples *which* scenes and *which* seed-buckets
within them are drawn (with replacement), exactly mirroring how episodes were
actually generated (paired seeds within a scene, independent scenes).

Reproducibility: pass `seed=<int>` for a deterministic run (e.g. in tests or
a frozen paper number); omit it to get fresh entropy from
`np.random.default_rng()` at call time. There is no hardcoded default seed --
"seeding controllable, not hardcoded" per Task 10.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np

Record = Mapping[str, Any]


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lo: float
    ci_hi: float
    ci_level: float
    n_boot: int          # number of bootstrap replicates that produced a finite value
    n_boot_requested: int
    samples: np.ndarray  # the bootstrap replicate values themselves, for diagnostics/plots


def _percentile_ci(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.nanpercentile(samples, 100 * alpha))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha)))
    return lo, hi


def flat_bootstrap(values: Sequence[Any], statistic_fn: Callable[[Sequence[Any]], float],
                    n_boot: int = 2000, ci: float = 0.95, seed: int | None = None,
                    rng: np.random.Generator | None = None) -> BootstrapResult:
    """Plain (non-hierarchical) bootstrap over a flat sequence of items: each
    replicate resamples `len(values)` items with replacement and applies
    `statistic_fn` to the resampled list. Useful when there is no
    scene/seed structure to respect (e.g. bootstrapping a single already-
    paired list of per-pair errors).
    """
    values = list(values)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap zero items")
    point_estimate = float(statistic_fn(values))
    g = rng if rng is not None else np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        idx = g.integers(0, n, size=n)
        resample = [values[i] for i in idx]
        try:
            samples[b] = float(statistic_fn(resample))
        except Exception:
            samples[b] = np.nan
    n_valid = int(np.sum(~np.isnan(samples)))
    lo, hi = _percentile_ci(samples, ci)
    return BootstrapResult(point_estimate=point_estimate, ci_lo=lo, ci_hi=hi, ci_level=ci,
                            n_boot=n_valid, n_boot_requested=n_boot, samples=samples)


def hierarchical_bootstrap(
    records: Sequence[Record],
    statistic_fn: Callable[[Sequence[Record]], float],
    scene_key: str | None = "scene",
    seed_key: str | None = "seed",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Hierarchical bootstrap: resample scenes with replacement, then within
    each resampled scene resample its (seed_key) buckets with replacement,
    concatenate everything drawn into one bootstrap sample, and evaluate
    `statistic_fn` on it. The point estimate is `statistic_fn(records)` on
    the *original*, unresampled data -- never the mean of the bootstrap
    replicates.

    `records` must each contain `scene_key` and `seed_key` (unless one or
    both are set to None, see below). Multiple records may share a
    (scene, seed) pair -- e.g. one record per policy for that episode seed --
    and they are always drawn together, preserving the paired-seed design.

    Set `scene_key=None` to skip the scene level entirely and treat all
    records as one scene (bootstrap only over seed-buckets); set
    `seed_key=None` to skip the seed level (each record is its own
    resampling unit within its scene). Setting both to None degenerates to
    a flat bootstrap over individual records -- use `flat_bootstrap` for
    that case directly if there is no hierarchy at all.
    """
    if not records:
        raise ValueError("cannot bootstrap zero records")

    point_estimate = float(statistic_fn(records))
    g = rng if rng is not None else np.random.default_rng(seed)

    # Build scene -> [seed-bucket, seed-bucket, ...] structure.
    scene_of: Callable[[Record], Hashable] = (lambda r: r[scene_key]) if scene_key else (lambda r: "__all__")
    seed_of: Callable[[Record], Hashable] = (lambda r: r[seed_key]) if seed_key else (lambda r: id(r))

    scenes: dict[Hashable, dict[Hashable, list[Record]]] = {}
    for r in records:
        s = scene_of(r)
        sd = seed_of(r)
        scenes.setdefault(s, {}).setdefault(sd, []).append(r)

    scene_ids = list(scenes.keys())
    n_scenes = len(scene_ids)
    # Precompute each scene's list of seed-buckets (each bucket is a list of records).
    scene_buckets = {s: list(seed_map.values()) for s, seed_map in scenes.items()}

    samples = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        resampled_scene_idx = g.integers(0, n_scenes, size=n_scenes)
        sample_records: list[Record] = []
        for si in resampled_scene_idx:
            buckets = scene_buckets[scene_ids[si]]
            n_buckets = len(buckets)
            resampled_bucket_idx = g.integers(0, n_buckets, size=n_buckets)
            for bi in resampled_bucket_idx:
                sample_records.extend(buckets[bi])
        try:
            samples[b] = float(statistic_fn(sample_records))
        except Exception:
            samples[b] = np.nan

    n_valid = int(np.sum(~np.isnan(samples)))
    lo, hi = _percentile_ci(samples, ci)
    return BootstrapResult(point_estimate=point_estimate, ci_lo=lo, ci_hi=hi, ci_level=ci,
                            n_boot=n_valid, n_boot_requested=n_boot, samples=samples)
