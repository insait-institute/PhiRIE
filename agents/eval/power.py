"""Simulation-based episode-budget power analysis (Task 10,
plan/10_PREDICTIVE_METRICS_AND_POWER.md step 4:
"Run simulation-based power analysis using plausible success rates and
observed variance.").

The concrete worked example this module is validated against is
docs/ROBOT.md's pi0.5/DROID-sim result: "from a 0.28 base nonzero rate,
detecting +0.15 needs ~158 episodes/arm (80% power, alpha 0.05); +0.25 needs
60; +0.40 needs 24" -- see tests/test_predictive_metrics.py for the anchor
test that reproduces these three numbers from this module within tolerance.

Method: rather than quoting a closed-form normal-approximation sample-size
formula, `required_episodes_two_proportion` runs an actual Monte Carlo
simulation of the two-proportion z-test (binomial draws for each arm, pooled-
variance test statistic, two-sided by default) at candidate episode counts
and searches for the smallest count whose empirical power meets the target.
This is what "simulation-based" means here: the reported episode count is
the number of Monte Carlo trials of two-proportion z-tests where a target
power fraction actually cleared alpha, not an asymptotic z-score formula
evaluated once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PowerResult:
    n_per_arm: int
    achieved_power: float
    base_rate: float
    effect_size: float
    target_power: float
    alpha: float
    two_sided: bool
    n_sims: int
    seed: int | None


def simulate_two_proportion_power(
    n_per_arm: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    n_sims: int = 4000,
    seed: int | None = None,
    two_sided: bool = True,
) -> float:
    """Empirical power of a two-proportion pooled-variance z-test (the same
    test as a 2x2 chi-square test of independence) to detect a difference
    between true rates `p1` and `p2` with `n_per_arm` episodes in each arm,
    estimated by drawing `n_sims` independent trial pairs and counting the
    fraction that reject H0: p1 == p2 at level `alpha`.
    """
    if n_per_arm < 1:
        raise ValueError(f"n_per_arm must be >= 1, got {n_per_arm}")
    for name, p in (("p1", p1), ("p2", p2)):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {p}")

    rng = np.random.default_rng(seed)
    x1 = rng.binomial(n_per_arm, p1, size=n_sims)
    x2 = rng.binomial(n_per_arm, p2, size=n_sims)
    phat1 = x1 / n_per_arm
    phat2 = x2 / n_per_arm
    pbar = (x1 + x2) / (2.0 * n_per_arm)
    se = np.sqrt(pbar * (1.0 - pbar) * (2.0 / n_per_arm))

    z = np.zeros(n_sims, dtype=float)
    nonzero = se > 0
    z[nonzero] = (phat1[nonzero] - phat2[nonzero]) / se[nonzero]
    # se == 0 only when pbar in {0, 1} (both arms all-success or all-failure):
    # no evidence of a difference either way, so z stays 0 (not rejected).

    if two_sided:
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        rejected = np.abs(z) > z_crit
    else:
        z_crit = stats.norm.ppf(1.0 - alpha)
        rejected = z > z_crit
    return float(np.mean(rejected))


def required_episodes_two_proportion(
    base_rate: float,
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    n_sims: int = 6000,
    seed: int | None = None,
    two_sided: bool = True,
    n_min: int = 2,
    n_max: int = 50_000,
) -> PowerResult:
    """Smallest per-arm episode count whose simulated power (see
    `simulate_two_proportion_power`) reaches `power`, for detecting a true
    rate change of `effect_size` from a `base_rate` baseline.

    Searches by exponential doubling to bracket the answer, then bisects.
    `seed` controls the Monte Carlo draws used at every candidate n during
    the search (deterministic for a fixed seed; omit for fresh entropy).
    """
    p1 = base_rate
    p2 = base_rate + effect_size
    if not (0.0 <= p1 <= 1.0):
        raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")
    if not (0.0 <= p2 <= 1.0):
        raise ValueError(f"base_rate + effect_size must be in [0, 1], got {p2}")

    def achieved(n: int) -> float:
        return simulate_two_proportion_power(n, p1, p2, alpha=alpha, n_sims=n_sims,
                                              seed=seed, two_sided=two_sided)

    lo = n_min
    hi = n_min
    hi_power = achieved(hi)
    while hi_power < power:
        lo = hi
        hi = hi * 2 if hi > 0 else 1
        if hi > n_max:
            raise RuntimeError(
                f"required n exceeds search bound n_max={n_max} "
                f"(base_rate={base_rate}, effect_size={effect_size}, power={power})"
            )
        hi_power = achieved(hi)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if achieved(mid) >= power:
            hi = mid
        else:
            lo = mid

    return PowerResult(n_per_arm=hi, achieved_power=achieved(hi), base_rate=base_rate,
                        effect_size=effect_size, target_power=power, alpha=alpha,
                        two_sided=two_sided, n_sims=n_sims, seed=seed)
