"""Regret metrics."""

from __future__ import annotations


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


def lai_robbins_lower_bound(env) -> float:
    """Lai-Robbins KL divergence constant; finite only for bandits with a
    KL(p_i || p*) defined. Caller may treat infinite as 'not available'.

    Returns the constant `C(env)` such that regret >= C(env) * ln T asymptotically.
    """
    import math
    pstar = env.optimal_mean
    c = 0.0
    for arm in range(env.n_arms):
        p = env.mean(arm)
        if p >= pstar:
            continue
        # Bernoulli KL(p || pstar) — works for Bernoulli; for Gaussians,
        # this approximation underestimates but stays informative.
        if 0.0 < p < 1.0 and 0.0 < pstar < 1.0:
            kl = p * math.log(p / pstar) + (1 - p) * math.log((1 - p) / (1 - pstar))
        else:
            kl = (pstar - p) ** 2 / 2.0  # Gaussian-ish fallback
        if kl > 0:
            c += (pstar - p) / kl
    return c
