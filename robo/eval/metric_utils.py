"""Small dependency-light utilities shared by the metric table generators."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def bootstrap_interval(values: Iterable[float], *, statistic: Callable = np.mean,
                       samples: int = 2000, seed: int = 0,
                       confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(statistic(array))
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    estimates = np.asarray([statistic(array[idx]) for idx in indices], dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(estimates, [alpha, 1.0 - alpha]))


def paired_bootstrap_delta(a: dict[str, float], b: dict[str, float], *,
                           samples: int = 4000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0, "delta": None, "ci95": [None, None]}
    deltas = np.asarray([b[i] - a[i] for i in ids], dtype=float)
    lo, hi = bootstrap_interval(deltas, samples=samples, seed=seed)
    return {"n": len(ids), "delta": float(np.mean(deltas)), "ci95": [lo, hi]}


def write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def paired_hierarchical_bootstrap(rows: list[dict], *, samples: int = 2000,
                                  seed: int = 42) -> dict:
    """Instance-macro B-A with paired layout/instance/reset resampling.

    Each row is one matched reset: layout_id, canonical_instance_id, reset_id,
    a, b. This estimates uncertainty over independent layouts, not episodes.
    Inputs must already be separated by task/protocol/sensor/policy/cohort.
    """
    clusters = {}
    seen = set()
    for row in rows:
        key = tuple(str(row[k]) for k in
                    ("layout_id", "canonical_instance_id", "reset_id"))
        if key in seen:
            raise ValueError(f"duplicate paired reset: {key}")
        seen.add(key)
        a, b = float(row["a"]), float(row["b"])
        if not np.isfinite([a, b]).all() or a not in (0, 1) or b not in (0, 1):
            raise ValueError("paired native success must be binary and finite")
        clusters.setdefault(key[0], {}).setdefault(key[1], []).append(b - a)
    result = {"layouts": len(clusters), "instances": sum(map(len, clusters.values())),
              "matched_resets": len(rows), "delta": None, "ci95": [None, None],
              "reference_only_success": sum(r["a"] == 1 and r["b"] == 0 for r in rows),
              "method_only_success": sum(r["a"] == 0 and r["b"] == 1 for r in rows),
              "both_success": sum(r["a"] == 1 and r["b"] == 1 for r in rows),
              "both_failure": sum(r["a"] == 0 and r["b"] == 0 for r in rows),
              "bootstrap_samples": samples, "seed": seed,
              "estimand": "instance_macro_method_minus_reference",
              "interval_scope": "descriptive; limited independent layout support"}
    if not rows:
        return result
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    result["delta"] = float(np.mean([np.mean(v) for c in clusters.values() for v in c.values()]))
    rng, layouts = np.random.default_rng(seed), sorted(clusters)
    estimates = []
    for _ in range(samples):
        instance_means = []
        for li in rng.integers(len(layouts), size=len(layouts)):
            instances = clusters[layouts[li]]
            keys = sorted(instances)
            for ii in rng.integers(len(keys), size=len(keys)):
                values = np.asarray(instances[keys[ii]])
                instance_means.append(float(np.mean(values[rng.integers(len(values), size=len(values))])))
        estimates.append(float(np.mean(instance_means)))
    result["ci95"] = [float(v) for v in np.quantile(estimates, [0.025, 0.975])]
    return result


def write_csv(path: str | Path, rows: list[dict], fieldnames=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
