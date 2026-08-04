"""Regret metrics."""

from __future__ import annotations

import math
from collections.abc import Callable

from .envs import BernoulliBandit, GaussianBandit


def cumulative_regret(env, arms_pulled: list[int], rewards: list[float]) -> list[float]:
    """Cumulative regret over time: optimal_mean * t - sum(rewards_so_far).

    Note: this uses observed reward sums (more practical) than expected reward;
    for theoretical regret use `cumulative_pseudo_regret` below.
    """
    cum_reward = 0.0
    out = []
    for t, r in enumerate(rewards, start=1):
        cum_reward += r
        out.append(env.optimal_mean * t - cum_reward)
    return out


def cumulative_pseudo_regret(env, arms_pulled: list[int]) -> list[float]:
    """Pseudo-regret: sum(optimal_mean - mean(arm_pulled)). Bound-friendly."""
    out, run = [], 0.0
    for arm in arms_pulled:
        run += env.optimal_mean - env.mean(arm)
        out.append(run)
    return out


def bernoulli_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) with the standard edge conventions.

    0*ln(0/x) = 0, and any nonzero mass placed where q has none gives +inf
    (q = 0 with p > 0, or q = 1 with p < 1).
    """
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("p and q must be in [0, 1]")
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return math.inf
    out = 0.0
    if p > 0.0:
        out += p * math.log(p / q)
    if p < 1.0:
        out += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return out


def gaussian_kl(mu_p: float, mu_q: float, sigma: float) -> float:
    """KL(N(mu_p, sigma^2) || N(mu_q, sigma^2)) = (mu_p - mu_q)^2 / (2 sigma^2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (mu_p - mu_q) ** 2 / (2.0 * sigma ** 2)


def lai_robbins_lower_bound(
    env, kl: Callable[[int], float] | None = None,
) -> float:
    """Lai-Robbins asymptotic lower-bound constant.

    Returns C(env) = sum over suboptimal arms of (mu* - mu_i) / KL_i, where
    KL_i is the divergence of arm i's reward distribution from the optimal
    arm's, so that any consistent policy has regret >= C(env) * ln T + o(ln T).

    The KL is chosen by the env's distribution family — exact Bernoulli KL for
    `BernoulliBandit` (arms whose KL is infinite, e.g. p* = 1, contribute
    zero: they are distinguished after finitely many pulls) and exact Gaussian
    KL for `GaussianBandit`, using each arm's own sigma.

    For any other env, pass `kl(arm) -> KL(arm dist || optimal dist)`.
    """
    if kl is None:
        if isinstance(env, BernoulliBandit):
            def kl(arm: int) -> float:
                return bernoulli_kl(env.mean(arm), env.optimal_mean)
        elif isinstance(env, GaussianBandit):
            def kl(arm: int) -> float:
                return gaussian_kl(env.mean(arm), env.optimal_mean,
                                   env.sigmas[arm])
        else:
            raise TypeError(
                f"no known KL for {type(env).__name__}; pass kl=... explicitly")
    pstar = env.optimal_mean
    c = 0.0
    for arm in range(env.n_arms):
        gap = pstar - env.mean(arm)
        if gap <= 0:
            continue
        d = kl(arm)
        if d <= 0:
            raise ValueError(f"kl({arm}) must be positive for a suboptimal arm")
        if math.isfinite(d):
            c += gap / d
    return c
